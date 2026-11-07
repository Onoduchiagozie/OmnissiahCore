# Branch and release workflow

Only main is permanent. Use a temporary branch for a change, review its pull
request, merge after tests and delete the merged branch. GPU/local and Oracle
settings are selected through private configuration, not machine-specific forks.

The root README.md contains the current setup, test, publish and rollback steps.
Historical oracle/lenovo branch tips will be preserved as tags before cleanup.
Lenovo has unpushed changes; reconcile them before declaring source migration complete.
