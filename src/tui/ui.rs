//! Rendering.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use super::app::{App, Entry, ToolStatus};
use super::text::{str_width, wrap_text};
use crate::types::ToolMode;

const MAX_INPUT_ROWS: usize = 6;
const COLLAPSED_LINES: usize = 6;

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// Push wrapped `text` with a first-line prefix and an indent for the rest.
fn push_wrapped(
    out: &mut Vec<Line<'static>>,
    text: &str,
    width: usize,
    first: Span<'static>,
    indent: usize,
    style: Style,
) {
    let pw = str_width(&first.content).max(indent);
    let inner = width.saturating_sub(pw).max(1);
    let mut first = Some(first);
    for l in wrap_text(text, inner) {
        let head = match first.take() {
            Some(f) => f,
            None => Span::raw(" ".repeat(pw)),
        };
        out.push(Line::from(vec![head, Span::styled(l, style)]));
    }
}

/// Split `**bold**` segments into spans. Unbalanced markers stay literal.
fn bold_spans(line: &str, base: Style) -> Vec<Span<'static>> {
    let parts: Vec<&str> = line.split("**").collect();
    if parts.len() < 3 || parts.len().is_multiple_of(2) {
        return vec![Span::styled(line.to_string(), base)];
    }
    parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_empty())
        .map(|(i, p)| {
            let st = if i % 2 == 1 {
                base.add_modifier(Modifier::BOLD)
            } else {
                base
            };
            Span::styled(p.to_string(), st)
        })
        .collect()
}

fn assistant_lines(text: &str, width: usize, out: &mut Vec<Line<'static>>) {
    let code_style = Style::default().fg(Color::Yellow);
    let mut in_code = false;
    for raw in text.split('\n') {
        let is_fence = raw.trim_start().starts_with("```");
        if is_fence {
            in_code = !in_code;
            for l in wrap_text(raw, width.saturating_sub(2).max(1)) {
                out.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(l, code_style.add_modifier(Modifier::DIM)),
                ]));
            }
        } else if in_code {
            for l in wrap_text(raw, width.saturating_sub(2).max(1)) {
                out.push(Line::from(vec![
                    Span::styled("▎ ", code_style),
                    Span::styled(l, code_style),
                ]));
            }
        } else {
            for l in wrap_text(raw, width.max(1)) {
                out.push(Line::from(bold_spans(&l, Style::default())));
            }
        }
    }
}

pub fn build_transcript(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    for (i, e) in app.entries.iter().enumerate() {
        if i > 0 {
            out.push(Line::raw(""));
        }
        match e {
            Entry::User(t) => push_wrapped(
                &mut out,
                t,
                width,
                Span::styled(
                    "> ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                2,
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Entry::Assistant { text, .. } => assistant_lines(text, width, &mut out),
            Entry::Info(t) => {
                for l in wrap_text(t, width.max(1)) {
                    out.push(Line::styled(l, dim()));
                }
            }
            Entry::Error(t) => push_wrapped(
                &mut out,
                t,
                width,
                Span::styled("✗ ", Style::default().fg(Color::Red)),
                2,
                Style::default().fg(Color::Red),
            ),
            Entry::Tool { title, status, .. } => {
                let (mark, color) = match status {
                    ToolStatus::Running => (app.spinner().to_string(), Color::Yellow),
                    ToolStatus::Done { ok: true, .. } => ("✓".to_string(), Color::Green),
                    ToolStatus::Done { ok: false, .. } => ("✗".to_string(), Color::Red),
                };
                let head = if matches!(status, ToolStatus::Running) {
                    format!("▶ {mark} ")
                } else {
                    format!("{mark} ")
                };
                push_wrapped(
                    &mut out,
                    title,
                    width,
                    Span::styled(head, Style::default().fg(color)),
                    2,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                );
                if let ToolStatus::Done { output, .. } = status {
                    let trimmed = output.trim_end();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let all: Vec<&str> = trimmed.lines().collect();
                    let shown = if app.expand_tools {
                        all.len()
                    } else {
                        all.len().min(COLLAPSED_LINES)
                    };
                    for l in &all[..shown] {
                        for w in wrap_text(l, width.saturating_sub(2).max(1)) {
                            out.push(Line::from(vec![Span::raw("  "), Span::styled(w, dim())]));
                        }
                    }
                    if shown < all.len() {
                        out.push(Line::styled(
                            format!("  (+{} lines, press Ctrl+O to expand)", all.len() - shown),
                            dim().add_modifier(Modifier::ITALIC),
                        ));
                    }
                }
            }
        }
    }
    out
}

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let width = area.width as usize;
    let (rows, cursor) = app.input.layout(width.saturating_sub(4).max(2));
    let shown_rows = rows.len().clamp(1, MAX_INPUT_ROWS);
    let input_h = shown_rows as u16 + 2;

    let [header, body, input_area, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(input_h.min(area.height.saturating_sub(3).max(3))),
        Constraint::Length(1),
    ])
    .areas(area);

    draw_header(f, app, header);
    draw_transcript(f, app, body);
    draw_input(f, app, input_area, &rows, cursor);
    draw_footer(f, app, footer);
    if app.current_approval().is_some() {
        f.set_cursor_position((area.x, area.y));
        draw_modal(f, app, area);
    }
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let model = if app.model.is_empty() {
        "(server default)"
    } else {
        &app.model
    };
    let mode = match app.tool_mode {
        ToolMode::Native => "native",
        ToolMode::Prompt => "prompt",
    };
    let mut spans = vec![
        Span::styled(
            " mistan ",
            Style::default().fg(Color::Black).bg(Color::Cyan),
        ),
        Span::raw(format!(" {model} ")),
        Span::styled(format!("{} ", app.base_url), dim()),
        Span::styled(format!("tools:{mode} "), dim()),
    ];
    if app.auto_approve {
        spans.push(Span::styled(
            " auto-approve ",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_transcript(f: &mut Frame, app: &mut App, area: Rect) {
    let lines = build_transcript(app, area.width as usize);
    let h = area.height as usize;
    let max_top = lines.len().saturating_sub(h);
    app.last_total = lines.len();
    app.last_height = h;
    let top = match app.scroll_top {
        None => max_top,
        Some(t) => t.min(max_top),
    };
    let visible: Vec<Line> = lines.into_iter().skip(top).take(h).collect();
    f.render_widget(Paragraph::new(visible), area);
    if top < max_top {
        let msg = format!(" ↓ {} more lines (End) ", max_top - top);
        let w = str_width(&msg) as u16;
        if area.width > w && area.height > 0 {
            let r = Rect::new(area.x + area.width - w, area.y + area.height - 1, w, 1);
            f.render_widget(
                Paragraph::new(Span::styled(
                    msg,
                    Style::default().fg(Color::Black).bg(Color::DarkGray),
                )),
                r,
            );
        }
    }
}

fn draw_input(f: &mut Frame, app: &App, area: Rect, rows: &[String], cursor: (usize, usize)) {
    let border = if app.busy {
        dim()
    } else {
        Style::default().fg(Color::Cyan)
    };
    let block = Block::default().borders(Borders::ALL).border_style(border);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let h = inner.height as usize;
    let off = (cursor.0 + 1).saturating_sub(h);
    let mut lines: Vec<Line> = Vec::new();
    if app.input.is_empty() {
        lines.push(Line::styled("Type a message…", dim()));
    } else {
        for r in rows.iter().skip(off).take(h) {
            lines.push(Line::raw(r.clone()));
        }
    }
    // 2-column gutter for the prompt marker
    let gutter = Rect::new(inner.x, inner.y, 2.min(inner.width), inner.height);
    let text_area = Rect::new(
        inner.x + gutter.width,
        inner.y,
        inner.width.saturating_sub(gutter.width),
        inner.height,
    );
    f.render_widget(
        Paragraph::new(Line::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        gutter,
    );
    f.render_widget(Paragraph::new(lines), text_area);
    if app.current_approval().is_none() {
        let cy = text_area.y + cursor.0.saturating_sub(off) as u16;
        let cx = text_area.x + cursor.1 as u16;
        if cx < text_area.x + text_area.width.max(1) && cy < text_area.y + text_area.height.max(1) {
            f.set_cursor_position((cx, cy));
        }
    }
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let line = if app.events_closed {
        Line::styled(
            " agent stopped - press any key to quit",
            Style::default().fg(Color::Red),
        )
    } else if app.current_approval().is_some() {
        Line::styled(
            " y yes  n no  a always (this session)",
            Style::default().fg(Color::Yellow),
        )
    } else if let Some((h, _)) = &app.hint {
        Line::styled(format!(" {h}"), Style::default().fg(Color::Yellow))
    } else if app.busy {
        Line::from(vec![
            Span::styled(
                format!(" {} ", app.spinner()),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(app.busy_label()),
            Span::styled("   Esc cancel  PgUp/PgDn scroll  Ctrl+O expand", dim()),
        ])
    } else {
        Line::styled(
            " Enter send  Alt+Enter newline  PgUp/PgDn scroll  Ctrl+O expand  /help  Ctrl+C quit",
            dim(),
        )
    };
    f.render_widget(Paragraph::new(line), area);
}

fn draw_modal(f: &mut Frame, app: &App, area: Rect) {
    let Some(p) = app.current_approval() else {
        return;
    };
    let w = area.width.saturating_sub(4).clamp(20, 80).min(area.width);
    let inner_w = w.saturating_sub(4) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for l in wrap_text(&p.title, inner_w) {
        lines.push(Line::styled(
            l,
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Yellow),
        ));
    }
    if !p.reason.trim().is_empty() {
        lines.push(Line::raw(""));
        for l in wrap_text(&p.reason, inner_w) {
            lines.push(Line::styled(l, dim()));
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled("[y]", Style::default().fg(Color::Green)),
        Span::raw(" yes  "),
        Span::styled("[n]", Style::default().fg(Color::Red)),
        Span::raw(" no  "),
        Span::styled("[a]", Style::default().fg(Color::Yellow)),
        Span::raw(" always (this session)"),
    ]));
    let queued = app.approvals.len().saturating_sub(1);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" Run this command? ");
    if queued > 0 {
        block = block.title_bottom(format!(" +{queued} more waiting "));
    }
    let h = (lines.len() as u16 + 2).min(area.height);
    let rect = Rect::new(
        area.x + (area.width.saturating_sub(w)) / 2,
        area.y + (area.height.saturating_sub(h)) / 2,
        w,
        h,
    );
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .alignment(Alignment::Left),
        rect,
    );
}
