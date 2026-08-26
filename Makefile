.PHONY: all fix check clean

# Auto-format and apply automated fixes
fix:
	npx prettier --write "docs/**/*.md" "README.md" "mkdocs.yml"
	npx markdownlint-cli "**/*.md" --config .markdownlint.json --fix

# Validate integrity without altering files
check:
	npx prettier --check "docs/**/*.md" "README.md" "mkdocs.yml"
	npx markdownlint-cli "**/*.md" --config .markdownlint.json
	mkdocs build --strict
	cog check

# Run fixes first, then validate the final build
all: fix check