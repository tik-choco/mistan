//! Display-width aware text wrapping (CJK safe).

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub fn str_width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

/// Wrap `s` (which may contain `\n`) to lines of at most `width` columns.
/// Prefers breaking at ASCII spaces; wide characters may break anywhere.
/// Blank lines are preserved.
pub fn wrap_text(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in s.split('\n') {
        wrap_line(line.trim_end_matches('\r'), width, &mut out);
    }
    out
}

fn wrap_line(line: &str, width: usize, out: &mut Vec<String>) {
    let width = width.max(1);
    let mut cur = String::new();
    let mut cw = 0usize;
    for ch in line.chars() {
        let w = ch.width().unwrap_or(0);
        if cw + w > width && cw > 0 {
            if ch == ' ' {
                out.push(std::mem::take(&mut cur));
                cw = 0;
                continue;
            }
            match cur.rfind(' ') {
                Some(pos) if pos + 1 < cur.len() && ch.is_ascii() => {
                    let tail = cur[pos + 1..].to_string();
                    cur.truncate(pos);
                    out.push(std::mem::take(&mut cur));
                    cw = str_width(&tail);
                    cur = tail;
                }
                _ => {
                    out.push(std::mem::take(&mut cur));
                    cw = 0;
                }
            }
        }
        cur.push(ch);
        cw += w;
    }
    out.push(cur);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_wraps_by_display_width() {
        let lines = wrap_text("あいうえお", 4);
        assert_eq!(lines, vec!["あい", "うえ", "お"]);
        // odd width: a wide char never straddles the edge
        let lines = wrap_text("あいうえお", 5);
        assert_eq!(lines, vec!["あい", "うえ", "お"]);
        for l in wrap_text("日本語とEnglishが混ざった文章です。", 7) {
            assert!(str_width(&l) <= 7, "{l:?}");
        }
    }

    #[test]
    fn ascii_word_wrap_and_blank_lines() {
        assert_eq!(
            wrap_text("hello world foo", 8),
            vec!["hello", "world", "foo"]
        );
        assert_eq!(wrap_text("a\n\nb", 10), vec!["a", "", "b"]);
    }
}
