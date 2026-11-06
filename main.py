"""Compatibility entry point. Local startup never creates a public tunnel."""
from manage import main

if __name__ == '__main__':
    raise SystemExit(main())
