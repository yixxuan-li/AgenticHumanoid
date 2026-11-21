"""Package the merged Atlas reconstruction and humanoid skill system."""
import sys
from package_skill_lab import main

if __name__ == "__main__":
    sys.argv.insert(1, "--merged")
    main()
