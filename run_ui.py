"""
Quick launcher for BrickDemand Streamlit UI
"""

import subprocess
import sys
import os
from pathlib import Path

os.environ['QDRANT_MODE'] = 'cloud'

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'plotly': 'Plotly',
        'sqlalchemy': 'SQLAlchemy',
        'langchain': 'LangChain'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name} found")
        except ImportError:
            print(f"❌ {name} not found")
            missing.append(name)
    
    return missing

def main():
    """Launch Streamlit app"""
    
    print("=" * 80)
    print("🧱 BrickDemand Inventory AI - Starting Web Interface")
    print("=" * 80)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required packages: {', '.join(missing)}")
        print("\n🔧 To install all dependencies, run:")
        print("   python install_dependencies.py")
        print("\nOr manually:")
        print("   pip install -r requirements_streamlit.txt")
        
        response = input("\n❓ Do you want to install now? (y/n): ").lower()
        if response == 'y':
            print("\n📦 Installing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
            print("✅ Dependencies installed")
        else:
            print("❌ Cannot start without dependencies. Exiting...")
            sys.exit(1)
    
    # Get app path
    app_path = Path(__file__).parent / "agent" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ App not found at: {app_path}")
        sys.exit(1)
    
    print(f"📂 App path: {app_path}")
    print("\n" + "=" * 80)
    print("🚀 Launching Streamlit...")
    print("=" * 80)
    print("\n💡 The app will open in your default browser at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the server\n")
    
    # Launch streamlit using Python module (works better on Windows)
    subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    main()

