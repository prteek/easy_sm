---
name: refresh
description: "Refresh and update CLAUDE.md and related documentation to reflect the current project state"
disable-model-invocation: true

---

When asked to refresh CLAUDE.md or update project documentation:

1. **Audit Current Project State**
   - List all files in `.claude/agents/` and verify which agents are active
   - List all files in `.claude/skills/` and identify all available skills
   - Check for any new integrations or external services configured in `.mcp.json`
   - Review `scripts/` directory for any validation or utility scripts
   - Check for any environment variables or configuration requirements

2. **Review Existing Documentation**
   - Read the current `CLAUDE.md` file
   - Identify any agents, skills, or integrations mentioned that no longer exist
   - Identify any agents, skills, or integrations that exist but are not documented

3. **Update CLAUDE.md**
   - Add or update sections for all active agents (read each agent's `.md` file)
   - Add or update sections for all available skills (read each skill's `SKILL.md` file)
   - Update external integrations section with current configurations
   - Update safety and validation section with any new scripts or checks
   - Update key patterns and conventions based on project git commits and practices
   - Maintain consistent formatting and structure
   - Ensure all cross-references to skill and agent files are accurate and include their full paths

4. **Update Related Documentation**
   - For each skill that has supplementary documentation (e.g., ADDITIONAL_INFO.md, QUESTIONS.md), verify it exists and is current
   - Update or add references to supplementary files as needed

5. **Commit Changes**
   - Stage all updated documentation files
   - Create a git commit with a clear message describing what was updated
   - Do NOT include *Co-Authored-By* in commit messages

6. **Communicate Changes**
   - Summarize what was updated in CLAUDE.md and any related files
   - Highlight any new features or agents that Claude should be aware of
   - Note any deprecated or removed features
