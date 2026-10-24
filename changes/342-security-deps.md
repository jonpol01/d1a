### Security

- filelock 4.2.4 and setuptools 84.2.1 (#342) close the open advisories on the locked dependencies: filelock's
  ReadWriteLock could be pointed at a planted lock database, and two Windows-only filelock issues and a setuptools
  Windows path escape in legacy wheel installs.
