//! Copy for connection settings, with the same keys in every supported locale.

const EN: &[(&str, &str)] = &[
    ("connection", "Connection"),
    ("enabled", "Enabled"),
    ("label", "Name"),
    ("yes", "on"),
    ("no", "off"),
    ("unassigned", "unassigned"),
    (
        "unavailable",
        "The assigned connection is disabled or missing.",
    ),
    (
        "rooms_in_mistl",
        "Configure Room providers in mistl and use backend = 'mistl'.",
    ),
    ("no_matches", "No matching models."),
    (
        "form_keys",
        "Ctrl+S save  Ctrl+N add HTTP  Ctrl+D delete connection  Esc cancel",
    ),
    (
        "delete_hint",
        "Press Ctrl+D again to delete this connection. References are kept.",
    ),
    ("model_source", "Models from"),
    ("invalid_url", "Enter a valid HTTP or HTTPS base URL."),
];

const JA: &[(&str, &str)] = &[
    ("connection", "接続先"),
    ("enabled", "有効"),
    ("label", "名前"),
    ("yes", "オン"),
    ("no", "オフ"),
    ("unassigned", "未設定"),
    (
        "unavailable",
        "割り当てた接続先が無効、または削除されています。",
    ),
    (
        "rooms_in_mistl",
        "Room の接続先は mistl で設定し、backend = 'mistl' を使用してください。",
    ),
    ("no_matches", "一致するモデルがありません。"),
    (
        "form_keys",
        "Ctrl+S 保存  Ctrl+N HTTP 追加  Ctrl+D 接続先削除  Esc キャンセル",
    ),
    (
        "delete_hint",
        "もう一度 Ctrl+D を押すと接続先を削除します。参照は保持されます。",
    ),
    ("model_source", "モデルの接続先"),
    (
        "invalid_url",
        "有効な HTTP または HTTPS の Base URL を入力してください。",
    ),
];

const ZH: &[(&str, &str)] = &[
    ("connection", "连接"),
    ("enabled", "启用"),
    ("label", "名称"),
    ("yes", "开启"),
    ("no", "关闭"),
    ("unassigned", "未设置"),
    ("unavailable", "指定的连接已禁用或已删除。"),
    (
        "rooms_in_mistl",
        "请在 mistl 中设置房间连接，并使用 backend = 'mistl'。",
    ),
    ("no_matches", "没有匹配的模型。"),
    (
        "form_keys",
        "Ctrl+S 保存  Ctrl+N 添加 HTTP  Ctrl+D 删除连接  Esc 取消",
    ),
    ("delete_hint", "再次按 Ctrl+D 删除此连接。模型引用将保留。"),
    ("model_source", "模型来源"),
    ("invalid_url", "请输入有效的 HTTP 或 HTTPS 地址。"),
];

pub fn get(key: &str) -> &'static str {
    let language = ["MISTAN_LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let strings = if language.starts_with("ja") {
        JA
    } else if language.starts_with("zh") {
        ZH
    } else {
        EN
    };
    strings
        .iter()
        .find(|(k, _)| *k == key)
        .expect("known locale key")
        .1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn locale_keys_and_placeholders_are_complete() {
        fn placeholders(s: &str) -> Vec<&str> {
            s.split('{')
                .skip(1)
                .map(|part| part.split('}').next().unwrap())
                .collect()
        }
        let reference: BTreeMap<_, _> = EN.iter().copied().collect();
        for locale in [EN, JA, ZH] {
            let strings: BTreeMap<_, _> = locale.iter().copied().collect();
            assert_eq!(strings.len(), locale.len(), "duplicate keys");
            assert_eq!(
                strings.keys().collect::<Vec<_>>(),
                reference.keys().collect::<Vec<_>>()
            );
            for (key, value) in strings {
                assert!(!value.trim().is_empty(), "{key}");
                assert_eq!(placeholders(value), placeholders(reference[key]), "{key}");
            }
        }
    }
}
