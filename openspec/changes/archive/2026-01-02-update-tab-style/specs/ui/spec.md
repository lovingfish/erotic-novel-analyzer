## MODIFIED Requirements

### Requirement: Tab Navigation Style
The tab navigation SHALL use an underline-based style for visual consistency.

#### Scenario: Tab container baseline
- **WHEN** the tab container is rendered
- **THEN** it displays a subtle bottom border as a visual baseline for all tabs

#### Scenario: Inactive tab appearance
- **WHEN** a tab is not selected
- **THEN** it has no bottom highlight and uses secondary text color

#### Scenario: Active tab appearance
- **WHEN** a tab is selected
- **THEN** it displays a primary-colored bottom border (underline) and uses primary text color

#### Scenario: Tab hover state
- **WHEN** user hovers over an inactive tab
- **THEN** it shows a subtle background highlight with rounded top corners
