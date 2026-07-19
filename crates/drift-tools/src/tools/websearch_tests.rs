use super::WebSearchTool;

#[test]
fn malformed_result_without_href_quote_does_not_panic() {
    // A truncated href must stay within the supplied HTML buffer.
    let html = r#"<a class="result__a" href="https://example.com"#;

    let results = WebSearchTool::parse_duckduckgo_results(html, 1);

    assert_eq!(results.len(), 1);
}

#[test]
fn malformed_result_without_closing_tag_does_not_panic() {
    // A missing closing anchor must use the remaining buffer as its title.
    let html = r#"<a href="https://example.com" class="result__a">Example"#;

    let results = WebSearchTool::parse_duckduckgo_results(html, 1);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].title, "Example");
}
