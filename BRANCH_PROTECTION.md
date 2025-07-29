# Branch Protection Setup

This document explains how to set up branch protection to prevent direct commits to the main branch.

## Local Git Hooks

Git hooks are included in this repository to prevent direct commits to the main branch.

### Setup After Cloning

**One command setup (recommended):**
```bash
git config core.hooksPath scripts/git-hooks
```

That's it! Git will now automatically use the hooks from the repository.

**Alternative: Use the installation script**
```bash
./scripts/install-git-hooks.sh
```

### Manual Installation

If you prefer to install manually:

**Method 1: Git core.hooksPath (recommended)**
```bash
# Tell Git to use hooks from scripts/git-hooks directory
git config core.hooksPath scripts/git-hooks
```

**Method 2: Copy hooks (traditional)**
```bash
# Copy hooks to your local .git/hooks directory
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit
cp scripts/git-hooks/pre-push .git/hooks/pre-push

# Make them executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push
```

### Hook Details

#### Pre-commit Hook
- **File**: `scripts/git-hooks/pre-commit` → `.git/hooks/pre-commit`
- **Purpose**: Prevents commits directly to the main branch
- **Action**: Blocks commit and provides helpful instructions

#### Pre-push Hook
- **File**: `scripts/git-hooks/pre-push` → `.git/hooks/pre-push`
- **Purpose**: Prevents pushes directly to the main branch
- **Action**: Blocks push and provides helpful instructions

**Note**: Git hooks are local to each repository clone and must be installed by each team member.

## GitHub Branch Protection Rules

For your GitHub upstream repository (`https://github.com/kahrendt/microWakeWord.git`):

1. Go to your GitHub repository
2. Navigate to **Settings** → **Branches**
3. Click **Add rule** or **Add branch protection rule**
4. Configure the following settings:

### Branch Name Pattern
```
main
```

### Protection Settings (Recommended)
- ✅ **Restrict pushes that create files larger than 100 MB**
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if you have CODEOWNERS file)
- ✅ **Require status checks to pass before merging**
  - Add any CI/CD checks you have configured
- ✅ **Require branches to be up to date before merging**
- ✅ **Require conversation resolution before merging**
- ✅ **Include administrators** (applies rules to repo admins too)
- ✅ **Allow force pushes** → **Specify who can force push** → Nobody
- ✅ **Allow deletions** → Disabled

## Local Git Server Branch Protection

For your local git server (`http://homeserver.lan:3001/kevin/microWakeWord.git`):

The specific steps depend on what Git server software you're running (Gitea, GitLab, etc.). Generally:

1. Access your Git server's web interface
2. Go to repository settings
3. Look for "Branch Protection" or "Protected Branches"
4. Add `main` as a protected branch with similar rules

## Testing the Protection

To test that the hooks work:

```bash
# This should be blocked
echo "test" > test.txt
git add test.txt
git commit -m "test commit"  # Should fail with error message

# This should work
git checkout -b test-branch
git commit -m "test commit"  # Should succeed
```

## Bypassing Protection (Emergency Use)

If you ever need to bypass the local hooks in an emergency:

```bash
# Bypass pre-commit hook
git commit --no-verify -m "emergency commit"

# Bypass pre-push hook  
git push --no-verify origin main
```

⚠️ **Warning**: Only use `--no-verify` in true emergencies as it defeats the purpose of the protection.

## Additional Recommendations

1. **Use conventional commit messages** for better project history
2. **Create feature branches** with descriptive names like `feature/add-new-model`
3. **Use pull requests** for all changes to maintain code review process
4. **Set up CI/CD** to run tests automatically on pull requests