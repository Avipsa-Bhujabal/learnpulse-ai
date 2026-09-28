# Release checklist

- [x] Complete tests and compile check
- [x] Protected-artifact signatures recorded and compared
- [x] API tested with synthetic requests
- [x] Detect-secrets and privacy scans completed
- [x] Dependency consistency checked
- [x] Public links audited
- [x] Restricted files excluded by Git and Docker policies
- [x] Add an MIT software license covering LearnPulse source code only
- [ ] Install Docker and execute container runtime validation
- [ ] Manually capture and inspect safe screenshots
- [ ] Initialize/review Git history, then manually commit and tag if approved
- [x] Conduct final human-style privacy and release review

Docker runtime validation remains conditional on CI or a host with Docker. Safe
screenshots were explicitly excluded after browser automation was unavailable;
no image was fabricated and no documentation link depends on them.
