"""``python -m wearpg <command>`` -> wearpg.cli.main."""

import sys

from wearpg.cli import main

if __name__ == "__main__":
    sys.exit(main())
