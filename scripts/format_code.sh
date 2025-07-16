#!/bin/bash
# Script to format all Python files with Black

echo "🔧 Formatting Python files with Black..."

# Install black if not installed
pip install black==23.12.0

# Format all Python files
black app/ scripts/ tests/

echo "✅ Black formatting complete!"

# Show the files that were reformatted
echo ""
echo "📝 Files reformatted:"
git diff --name-only

echo ""
echo "💡 Now you can commit these changes"