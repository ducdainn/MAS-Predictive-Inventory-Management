"""
SQL Agent Upgrade Script
Upgrades SQLAgent in improved_mas.py to use the advanced V2 implementation
"""

import re
import shutil
import sys
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def backup_file(filepath):
    """Create a backup of the original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def read_file(filepath):
    """Read file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def find_sql_agent_section(content):
    """Find the SQLAgent class section to replace."""
    # Find start of SQLAgent class
    class_pattern = r'# ={70,}\n# SQL AGENT\n# ={70,}\n\nclass SQLAgent:'
    match = re.search(class_pattern, content)
    
    if not match:
        # Try simpler pattern
        match = re.search(r'class SQLAgent:', content)
    
    if not match:
        raise ValueError("Could not find SQLAgent class definition")
    
    start_pos = match.start()
    
    # Find end of SQLAgent class (next class definition or section marker)
    end_pattern = r'\n# ={70,}\n# [A-Z]'
    end_match = re.search(end_pattern, content[start_pos + 100:])
    
    if not end_match:
        raise ValueError("Could not find end of SQLAgent class")
    
    end_pos = start_pos + 100 + end_match.start()
    
    return start_pos, end_pos

def update_orchestrator_init(content):
    """Update OrchestratorAgent.__init__ to pass db_manager to SQLAgent."""
    # Find the SQLAgent initialization line
    pattern = r'self\.sql_agent = SQLAgent\(llm_provider, self\.schema_agent\)'
    replacement = r'self.sql_agent = SQLAgent(llm_provider, self.schema_agent, db_manager)'
    
    if pattern in content:
        content = re.sub(pattern, replacement, content)
        print("✅ Updated OrchestratorAgent.__init__ to pass db_manager")
    else:
        print("⚠️  Could not find SQLAgent initialization in OrchestratorAgent.__init__")
        print("   Please manually add db_manager parameter to SQLAgent initialization")
    
    return content

def main():
    print("=" * 80)
    print("SQL Agent Upgrade Script")
    print("=" * 80)
    print()
    
    # Paths
    mas_file = "agent/improved_mas.py"
    new_agent_file = "agent/sql_agent_v2.py"
    
    print("📖 Step 1: Reading files...")
    
    try:
        mas_content = read_file(mas_file)
        new_agent_content = read_file(new_agent_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you're in the project root directory")
        return
    
    print("✅ Files read successfully")
    print()
    
    print("💾 Step 2: Creating backup...")
    backup_path = backup_file(mas_file)
    print()
    
    print("🔍 Step 3: Locating SQLAgent class...")
    try:
        start_pos, end_pos = find_sql_agent_section(mas_content)
        print(f"✅ Found SQLAgent class at positions {start_pos}-{end_pos}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   Manual replacement required")
        return
    
    print()
    print("✂️  Step 4: Extracting new SQLAgent implementation...")
    
    # Extract only the SQLAgent class from new file (skip docstring at top)
    class_start = new_agent_content.find('class SQLAgent:')
    if class_start == -1:
        print("❌ Error: Could not find SQLAgent class in sql_agent_v2.py")
        return
    
    # Find section header to preserve
    header = """# ============================================================================
# SQL AGENT - IMPROVED VERSION WITH ADVANCED VALIDATION
# ============================================================================

"""
    
    new_agent_class = header + new_agent_content[class_start:]
    print(f"✅ Extracted {len(new_agent_class)} characters")
    print()
    
    print("🔧 Step 5: Replacing SQLAgent class...")
    
    # Replace old SQLAgent with new one
    updated_content = (
        mas_content[:start_pos] +
        new_agent_class +
        mas_content[end_pos:]
    )
    
    print("✅ SQLAgent class replaced")
    print()
    
    print("🔧 Step 6: Updating OrchestratorAgent initialization...")
    updated_content = update_orchestrator_init(updated_content)
    print()
    
    print("💾 Step 7: Writing updated file...")
    write_file(mas_file, updated_content)
    print(f"✅ File written: {mas_file}")
    print()
    
    print("=" * 80)
    print("✅ UPGRADE COMPLETE!")
    print("=" * 80)
    print()
    print("Summary of changes:")
    print("  ✅ SQLAgent class replaced with V2 implementation")
    print("  ✅ Added multi-retry mechanism (3 attempts)")
    print("  ✅ Added dry-run validation with EXPLAIN")
    print("  ✅ Added intelligent error recovery")
    print("  ✅ Added template-based fast path")
    print("  ✅ Added graceful degradation")
    print()
    print(f"Backup file: {backup_path}")
    print()
    print("Next steps:")
    print("  1. Test the updated agent:")
    print("     python -c \"from agent.improved_mas import initialize_system; orch = initialize_system()\"")
    print("  2. Run the UI:")
    print("     python run_ui.py")
    print()
    print("If issues occur, restore from backup:")
    print(f"     cp {backup_path} {mas_file}")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        print()
        print("Please perform manual upgrade:")
        print("  1. Open agent/improved_mas.py")
        print("  2. Find class SQLAgent (around line 680)")
        print("  3. Replace entire class with content from agent/sql_agent_v2.py")
        print("  4. In OrchestratorAgent.__init__, change:")
        print("     self.sql_agent = SQLAgent(llm_provider, self.schema_agent)")
        print("     to:")
        print("     self.sql_agent = SQLAgent(llm_provider, self.schema_agent, db_manager)")
        print()

