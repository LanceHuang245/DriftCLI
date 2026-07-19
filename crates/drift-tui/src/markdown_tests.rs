use super::*;

fn lines_to_string(lines: &[Line]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn unordered_list() {
    let lines = render_markdown("- Item 1\n- Item 2\n", 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("- Item 1"), "text was: {text}");
    assert!(text.contains("- Item 2"), "text was: {text}");
}

#[test]
fn ordered_list() {
    let lines = render_markdown("1. First\n2. Second\n", 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("1. First"), "text was: {text}");
    assert!(text.contains("2. Second"), "text was: {text}");
}

#[test]
fn tight_unordered_list() {
    let lines = render_markdown("- A\n- B\n- C\n", 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("- A"), "text was: {text}");
    assert!(text.contains("- B"), "text was: {text}");
    assert!(text.contains("- C"), "text was: {text}");
}

#[test]
fn tight_ordered_list() {
    let lines = render_markdown("1. Alpha\n2. Beta\n3. Gamma\n", 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("1. Alpha"), "text was: {text}");
    assert!(text.contains("2. Beta"), "text was: {text}");
    assert!(text.contains("3. Gamma"), "text was: {text}");
}

#[test]
fn nested_list() {
    let lines = render_markdown("- Outer\n  - Inner\n", 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("- Outer"), "text was: {text}");
    assert!(text.contains("- Inner"), "text was: {text}");
}

#[test]
fn basic_table() {
    let md = "| A | B |\n| - | - |\n| 1 | 2 |\n";
    let lines = render_markdown(md, 80);
    // Should produce bordered table rows (at least top border, header, separator)
    assert!(
        lines.len() >= 4,
        "expected at least 4 lines for bordered table"
    );
    let text = lines_to_string(&lines);
    assert!(text.contains("┌"), "table top border missing: {text}");
    assert!(text.contains("├"), "table header sep missing: {text}");
    assert!(text.contains("└"), "table bottom border missing: {text}");
    assert!(text.contains("A"), "header A missing: {text}");
    assert!(text.contains("B"), "header B missing: {text}");
    assert!(text.contains("1"), "cell 1 missing: {text}");
    assert!(text.contains("2"), "cell 2 missing: {text}");
}

#[test]
fn table_with_styled_cell() {
    let md = "| **Bold** | Normal |\n| - | - |\n| data | more |\n";
    let lines = render_markdown(md, 80);
    let text = lines_to_string(&lines);
    assert!(text.contains("Bold"), "bold cell missing: {text}");
    assert!(text.contains("Normal"), "normal cell missing: {text}");
}

#[test]
fn table_not_separate_lines() {
    let md = "| X | Y |\n| - | - |\n| a | b |\n";
    let lines = render_markdown(md, 80);
    let text = lines_to_string(&lines);
    // Should NOT have standalone "X" or "Y" lines — they should be in the table row
    let line_list: Vec<&str> = text.lines().collect();
    for line in &line_list {
        assert!(!line.trim().eq("X"), "found orphan X line: {text}");
        assert!(!line.trim().eq("Y"), "found orphan Y line: {text}");
    }
}

#[test]
fn table_multi_body_rows_have_separators() {
    let md = "| A | B |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |\n";
    let lines = render_markdown(md, 80);
    let text = lines_to_string(&lines);
    // Count occurrences of the separator border ├
    let sep_count = text.chars().filter(|&c| c == '├').count();
    assert_eq!(
        sep_count, 2,
        "expected 2 separators (header-body + body-body), got {sep_count}: {text}"
    );
}
