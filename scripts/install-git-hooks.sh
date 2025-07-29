#!/bin/bash
#
# Install Git hooks to prevent commits to main branch
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$SCRIPT_DIR/git-hooks"
GIT_HOOKS_DIR=".git/hooks"

echo "🔧 Installing Git hooks for branch protection..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository root directory"
    echo "   Please run this script from the repository root"
    exit 1
fi

# Check if hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo "❌ Error: Hooks directory not found at $HOOKS_DIR"
    exit 1
fi

# Install pre-commit hook
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✅ Installed pre-commit hook"
else
    echo "⚠️  Warning: pre-commit hook not found"
fi

# Install pre-push hook
if [ -f "$HOOKS_DIR/pre-push" ]; then
    cp "$HOOKS_DIR/pre-push" "$GIT_HOOKS_DIR/pre-push"
    chmod +x "$GIT_HOOKS_DIR/pre-push"
    echo "✅ Installed pre-push hook"
else
    echo "⚠️  Warning: pre-push hook not found"
fi

echo ""
echo "🎉 Git hooks installation complete!"
echo ""
echo "🚫 Direct commits and pushes to 'main' branch are now blocked"
echo "📋 Use feature branches and pull requests for all changes"
echo ""
echo "To test the protection:"
echo "  # This should fail:"
echo "  git checkout main"
echo "  echo 'test' > test.txt && git add test.txt && git commit -m 'test'"
echo ""
echo "  # This should work:"
echo "  git checkout -b feature/test-branch"
echo "  git commit -m 'test'"