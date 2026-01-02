## ADDED Requirements

### Requirement: Centered Navbar Layout
The navbar SHALL display the application title and control buttons (settings, theme toggle) centered horizontally to align with the main content area.

#### Scenario: Navbar renders centered
- **WHEN** the page loads
- **THEN** the navbar content (title + settings + theme toggle) is horizontally centered
- **AND** aligns with the `max-w-4xl` main content container

### Requirement: Application Title
The application title SHALL be "涩涩小说分析器" in both the HTML `<title>` tag and the navbar display.

#### Scenario: Title displays correctly
- **WHEN** the page loads
- **THEN** the browser tab shows "涩涩小说分析器"
- **AND** the navbar displays "涩涩小说分析器"
