//! Download and install the `mistl` binary from its GitHub releases.
//!
//! The release workflow publishes the raw binary per target
//! (`mistl-v<version>-<target>[.exe]`) next to `SHA256SUMS.txt`. The binary is
//! verified against that checksum list before it is moved into place. (The
//! checksum list is not signature-checked here; `mistl update` does that.)

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde_json::Value;
use sha2::{Digest, Sha256};

const REPO: &str = "tik-choco/mistl";
/// Hard cap for the downloaded binary.
const MAX_BINARY_BYTES: usize = 256 * 1024 * 1024;
const MAX_SUMS_BYTES: usize = 1024 * 1024;

/// Rust target triple of the running platform that mistl releases cover.
pub fn target_triple() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Some("aarch64-unknown-linux-gnu"),
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("macos", "x86_64") => Some("x86_64-apple-darwin"),
        _ => None,
    }
}

/// Per-user install location: the one `mistl install` uses on Windows,
/// `~/.local/bin/mistl` elsewhere.
pub fn install_path() -> Option<PathBuf> {
    if cfg!(windows) {
        dirs::data_local_dir().map(|d| d.join("Programs").join("mistl").join("mistl.exe"))
    } else {
        dirs::home_dir().map(|d| d.join(".local").join("bin").join("mistl"))
    }
}

/// A release asset chosen for this platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub tag: String,
    pub asset_name: String,
    pub asset_url: String,
    pub sums_url: String,
}

/// Pick the raw-binary asset for `target` and the checksum list out of a
/// GitHub "latest release" JSON document.
pub fn pick_release(release: &Value, target: &str) -> Result<Release> {
    let tag = release["tag_name"]
        .as_str()
        .ok_or_else(|| anyhow!("release JSON has no tag_name"))?
        .to_string();
    let suffix = format!(
        "-{target}{}",
        if target.contains("windows") {
            ".exe"
        } else {
            ""
        }
    );
    let assets = release["assets"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let find = |pred: &dyn Fn(&str) -> bool| {
        assets.iter().find_map(|a| {
            let name = a["name"].as_str()?;
            let url = a["browser_download_url"].as_str()?;
            pred(name).then(|| (name.to_string(), url.to_string()))
        })
    };
    let (asset_name, asset_url) = find(&|n| n.starts_with("mistl-v") && n.ends_with(&suffix))
        .ok_or_else(|| anyhow!("release {tag} has no binary for {target}"))?;
    let (_, sums_url) = find(&|n| n == "SHA256SUMS.txt")
        .ok_or_else(|| anyhow!("release {tag} has no SHA256SUMS.txt; refusing to install"))?;
    Ok(Release {
        tag,
        asset_name,
        asset_url,
        sums_url,
    })
}

/// Look up `name` in a `sha256sum`-style listing (`<hex>  <name>` / `<hex> *<name>`).
pub fn expected_sha256(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (hash, file) = line.trim().split_once(char::is_whitespace)?;
        let file = file.trim().trim_start_matches('*');
        (file == name && hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| hash.to_ascii_lowercase())
    })
}

fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(concat!("mistan/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(600))
        .build()?)
}

async fn get_bytes(http: &reqwest::Client, url: &str, max: usize) -> Result<Vec<u8>> {
    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow!("request failed: {}", e.without_url()))?;
    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {} from {}", status.as_u16(), host_of(url));
    }
    let mut out = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow!("download failed: {}", e.without_url()))?;
        if out.len() + chunk.len() > max {
            bail!("download exceeds {} MiB", max / 1024 / 1024);
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

fn host_of(url: &str) -> &str {
    url.split('/').nth(2).unwrap_or(url)
}

/// Resolve the newest release for this platform.
pub async fn latest_release() -> Result<Release> {
    let target = target_triple().ok_or_else(|| {
        anyhow!(
            "no prebuilt mistl for {}-{}; build it from source",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let http = http_client()?;
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let body = get_bytes(&http, &url, 4 * 1024 * 1024)
        .await
        .context("looking up the latest mistl release")?;
    let json: Value = serde_json::from_slice(&body).context("parsing the release JSON")?;
    pick_release(&json, target)
}

/// Download `release`, verify its checksum, and install it at `dest`.
/// Returns the installed path.
pub async fn install_release(release: &Release, dest: &Path) -> Result<PathBuf> {
    let http = http_client()?;
    let sums = get_bytes(&http, &release.sums_url, MAX_SUMS_BYTES)
        .await
        .context("downloading SHA256SUMS.txt")?;
    let sums = String::from_utf8_lossy(&sums);
    let expected = expected_sha256(&sums, &release.asset_name)
        .ok_or_else(|| anyhow!("SHA256SUMS.txt has no entry for {}", release.asset_name))?;
    let bytes = get_bytes(&http, &release.asset_url, MAX_BINARY_BYTES)
        .await
        .with_context(|| format!("downloading {}", release.asset_name))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        bail!(
            "checksum mismatch for {} (expected {expected}, got {actual})",
            release.asset_name
        );
    }
    write_executable(dest, &bytes)?;
    Ok(dest.to_path_buf())
}

fn write_executable(dest: &Path, bytes: &[u8]) -> Result<()> {
    let dir = dest
        .parent()
        .ok_or_else(|| anyhow!("invalid install path {}", dest.display()))?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    // Unique per process and exclusive, so concurrent installs cannot mix bytes.
    let tmp = dest.with_extension(format!("download.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        // On Windows a running mistl.exe cannot be replaced.
        return Err(e).with_context(|| {
            format!(
                "replacing {} (stop a running mistl first, e.g. `mistl daemon stop`)",
                dest.display()
            )
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn release_json() -> Value {
        json!({
            "tag_name": "v1.2.3",
            "assets": [
                { "name": "mistl-v1.2.3-x86_64-pc-windows-msvc.zip", "browser_download_url": "https://h/zip" },
                { "name": "mistl-v1.2.3-x86_64-pc-windows-msvc.exe", "browser_download_url": "https://h/win" },
                { "name": "mistl-v1.2.3-x86_64-unknown-linux-gnu", "browser_download_url": "https://h/linux" },
                { "name": "mistl-v1.2.3-x86_64-unknown-linux-gnu.tar.gz", "browser_download_url": "https://h/tgz" },
                { "name": "SHA256SUMS.txt", "browser_download_url": "https://h/sums" },
            ]
        })
    }

    #[test]
    fn picks_raw_binary_per_target() {
        let win = pick_release(&release_json(), "x86_64-pc-windows-msvc").unwrap();
        assert_eq!(win.asset_url, "https://h/win");
        assert_eq!(win.asset_name, "mistl-v1.2.3-x86_64-pc-windows-msvc.exe");
        assert_eq!(win.sums_url, "https://h/sums");
        let linux = pick_release(&release_json(), "x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(linux.asset_url, "https://h/linux");
        assert!(
            pick_release(&release_json(), "aarch64-apple-darwin")
                .unwrap_err()
                .to_string()
                .contains("no binary")
        );
    }

    #[test]
    fn refuses_release_without_checksums() {
        let mut j = release_json();
        j["assets"].as_array_mut().unwrap().pop();
        assert!(
            pick_release(&j, "x86_64-pc-windows-msvc")
                .unwrap_err()
                .to_string()
                .contains("SHA256SUMS.txt")
        );
    }

    #[test]
    fn parses_checksum_listing() {
        let h = "a".repeat(64);
        let sums = format!(
            "{h}  mistl-v1-x\n{}  *other\nnot-a-hash  bad\n",
            "B".repeat(64)
        );
        assert_eq!(expected_sha256(&sums, "mistl-v1-x"), Some(h));
        assert_eq!(expected_sha256(&sums, "other"), Some("b".repeat(64)));
        assert_eq!(expected_sha256(&sums, "bad"), None);
        assert_eq!(expected_sha256(&sums, "missing"), None);
    }

    #[test]
    fn write_executable_replaces_file() {
        let dir = std::env::temp_dir().join(format!("mistan-install-test-{}", std::process::id()));
        let dest = dir.join("nested").join("mistl-test");
        write_executable(&dest, b"one").unwrap();
        write_executable(&dest, b"two").unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"two");
        assert!(
            !dest
                .with_extension(format!("download.{}", std::process::id()))
                .exists()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
