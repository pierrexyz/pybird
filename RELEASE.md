# Release Guide

This guide explains how to release a new version of PyBird-LSS to PyPI.

## Prerequisites

Before you can release, you need to set up PyPI credentials:

### 1. PyPI Account Setup
- Create accounts on [PyPI](https://pypi.org) and [Test PyPI](https://test.pypi.org)
- Enable two-factor authentication (2FA) on both accounts
- Create API tokens with "Entire account (all projects)" scope

### 2. GitHub Secrets
Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `PYPI_API_TOKEN`: Your PyPI API token
- `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

## Release Process

### Option 1: Automated Release (Recommended)

1. **Run the release script:**
   ```bash
   python scripts/release.py 0.3.1
   ```

   This will:
   - Update the version in `pyproject.toml`
   - Create a git commit
   - Create and push a git tag (e.g., `v0.3.1`)
   - Trigger the GitHub Action

2. **Monitor the release:**
   - Check the [Actions tab](https://github.com/your-repo/actions) in your GitHub repository
   - The action will first publish to Test PyPI, then to PyPI
   - Both jobs must succeed for the release to complete

### Option 2: Manual Release

1. **Update version in `pyproject.toml`:**
   ```toml
   version = "0.3.1"
   ```

2. **Create and push a git tag:**
   ```bash
   git add pyproject.toml
   git commit -m "Release version 0.3.1"
   git tag v0.3.1
   git push origin master
   git push origin v0.3.1
   ```

## What Happens During Release

1. **Test PyPI Job:**
   - Builds the package
   - Publishes to Test PyPI
   - Tests installation from Test PyPI

2. **PyPI Job:**
   - Only runs if Test PyPI succeeds
   - Builds the package again
   - Publishes to PyPI

## Version Format

Use semantic versioning: `X.Y.Z`
- `X`: Major version (breaking changes)
- `Y`: Minor version (new features, backward compatible)
- `Z`: Patch version (bug fixes, backward compatible)

Examples: `0.3.0`, `0.3.1`, `1.0.0`

## Troubleshooting

### Common Issues

1. **Action fails on Test PyPI:**
   - Check that `TEST_PYPI_API_TOKEN` is set correctly
   - Verify your Test PyPI account has the correct permissions

2. **Action fails on PyPI:**
   - Check that `PYPI_API_TOKEN` is set correctly
   - Verify your PyPI account has the correct permissions
   - Ensure the version doesn't already exist on PyPI

3. **Package name conflicts:**
   - The package name `pybird-lss` must be unique on PyPI
   - Check if the name is already taken

### Manual Fallback

If the automated release fails, you can manually publish:

```bash
# Build the package
python -m build

# Publish to Test PyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for all sensitive credentials
- The tokens have "Entire account" scope - they can publish any package under your account 