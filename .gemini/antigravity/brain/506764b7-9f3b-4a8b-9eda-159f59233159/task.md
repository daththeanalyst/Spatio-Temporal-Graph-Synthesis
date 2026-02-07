# Task List

- [ ] Diagnose Experience section rendering issue <!-- id: 0 -->
    - [x] Inspect template data structure (`TEMPLATE_ULTRA_KINETIC.ts`) <!-- id: 1 -->
    - [x] Locate rendering component for Experience section <!-- id: 2 -->
    - [x] specific check for `experience` vs `work` property naming mismatch <!-- id: 3 -->
- [ ] Fix Experience section rendering <!-- id: 4 -->
    - [x] Fix property usage in all affected templates (Kinetic, Mesh, Futurism, Pastel, Bento, Quant, Neural, Lens, Horizontal Story)
    - [x] Change `{{this.date}}` to `{{this.dates}}` in Experience section
    - [x] Change `{{this.date}}` to `{{this.year}}` in Education section
- [x] Create `TEMPLATE_DEVELOPMENT_GUIDE.md` for future consistency
- [x] Verify fixes by generating/previewing templates (User Verification)ch if present
- [ ] Verify fix <!-- id: 5 -->
