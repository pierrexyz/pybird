#!/bin/bash

# Script to build PyBird documentation locally
# Usage: ./build_docs.sh

set -e  # Exit on any error

echo "Building PyBird documentation..."

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo "Error: Makefile not found. Please run this script from the docs/ directory."
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for pandoc (required for notebook rendering)
if ! command -v pandoc &> /dev/null; then
    echo "Warning: pandoc not found. Notebook rendering may fail."
    echo "To install pandoc:"
    echo "  Ubuntu/Debian: sudo apt-get install pandoc"
    echo "  macOS: brew install pandoc"
    echo "  Or: conda install pandoc"
    echo ""
    echo "Continuing without pandoc..."
    
    # Create a temporary conf.py without nbsphinx
    echo "Creating temporary configuration without notebook support..."
    cp source/conf.py source/conf_backup.py
    
    # Comment out nbsphinx in conf.py
    sed -i 's/extensions.append('\''nbsphinx'\'')/# extensions.append('\''nbsphinx'\'')/g' source/conf.py
    sed -i 's/NBSphinx_AVAILABLE = True/NBSphinx_AVAILABLE = False/g' source/conf.py
fi

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Generate API documentation
echo "Generating API documentation..."
sphinx-apidoc -f -M -e -o source/api ../pybird

# Build HTML documentation
echo "Building HTML documentation..."
if make html; then
    echo "Documentation built successfully!"
    echo "You can view it by opening docs/build/html/index.html in your browser."
    
    # Restore original conf.py if we modified it
    if [ -f "source/conf_backup.py" ]; then
        mv source/conf_backup.py source/conf.py
        echo "Configuration restored."
    fi
else
    echo "Documentation build failed!"
    
    # Restore original conf.py if we modified it
    if [ -f "source/conf_backup.py" ]; then
        mv source/conf_backup.py source/conf.py
        echo "Configuration restored."
    fi
    
    exit 1
fi 