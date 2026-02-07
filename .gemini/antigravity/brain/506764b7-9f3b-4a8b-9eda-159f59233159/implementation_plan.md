# Implementation Plan - Fix Experience Section Rendering

## Problem
The Experience section in the Ultra templates is not rendering the experience information.

## Proposed Changes

### Diagnosis
1.  **Check Data Structure:**#### [MODIFY] [TEMPLATE_ULTRA_KINETIC.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_KINETIC.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_MESH.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_MESH.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_FUTURISM.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_FUTURISM.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_PASTEL.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_PASTEL.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_BENTO_GRID.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_BENTO_GRID.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_QUANT.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_QUANT.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_NEURAL.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_NEURAL.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_ULTRA_LENS.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_ULTRA_LENS.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
#### [MODIFY] [TEMPLATE_HORIZONTAL_STORY.ts](file:///c:/Users/Dath/OneDrive/Desktop/AntiGravity%20Stuff/DataPorfolio-version1/src/templates/TEMPLATE_HORIZONTAL_STORY.ts) - **[FIXED]** Experience (`dates`) & Education (`year`).
2.  **Check Rendering Logic:** Identify the component responsible for rendering the Ultra templates (likely `TemplateWireframe.tsx` or a specific Ultra renderer) and check how it accesses the experience data.
    *   *Hypothesis:* There might be a property name mismatch (e.g., `experience` vs `workHistory` vs `jobs`).

### Fix
*   Update the rendering component to use the correct property path.
*   OR update the template data structure to match the renderer.

## Verification
*   Manual verification by checking the component code.
*   User verification (since I cannot run the full app easily to see the UI, I will rely on code correctness and user confirmation).
