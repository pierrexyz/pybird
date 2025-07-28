#!/usr/bin/env python3
"""
Robust documentation builder for PyBird.
This script builds the documentation without relying on external sphinx-build.
"""

import os
import sys
import shutil
from pathlib import Path

def build_docs():
    """Build the documentation using Python sphinx module."""
    
    # Set up paths
    docs_dir = Path(__file__).parent
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"
    
    print("Building PyBird documentation...")
    print(f"Source directory: {source_dir}")
    print(f"Build directory: {build_dir}")
    
    # Clean build directory
    if build_dir.exists():
        print("Cleaning previous build...")
        shutil.rmtree(build_dir)
    
    # Try to import and use Sphinx
    try:
        from sphinx.cmd.build import build_main
        print("Found Sphinx module, starting build...")
        
        # Build HTML documentation
        argv = [
            str(source_dir),  # source directory
            str(build_dir / "html"),  # output directory  
            "-b", "html",  # builder
            "-E",  # rebuild all files
            "-a",  # build all files
            "-v",  # verbose
        ]
        
        result = build_main(argv)
        
        if result == 0:
            print("✅ Documentation build successful!")
            print(f"📄 Documentation is available at: {build_dir / 'html' / 'index.html'}")
            return True
        else:
            print(f"❌ Documentation build failed with exit code: {result}")
            return False
            
    except ImportError:
        print("❌ Sphinx not available. Please install sphinx:")
        print("   pip install sphinx sphinx-rtd-theme")
        return False
    except Exception as e:
        print(f"❌ Build failed with error: {e}")
        return False

if __name__ == "__main__":
    success = build_docs()
    sys.exit(0 if success else 1) 