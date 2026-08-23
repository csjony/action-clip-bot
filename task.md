# Task Checklist: Splitting & Publishing Fixes

- [x] Modify `src/publish/base.py` to allow optional `horizontal` path
- [x] Modify `src/publish/coordinator.py` to include `facebook` in `shorts_platforms` and set `horizontal=None` on split parts
- [x] Modify `src/publish/native/meta.py` to check `assets.horizontal` existence before using it
- [x] Modify `src/publish/native/youtube.py` to check `assets.horizontal` existence before using it
- [x] Verify imports and code correctness via test execution
- [x] Run dry-run test to verify the full flow (generation -> composition -> splitting -> publishing)

---

# Next Session: Cinematic Photorealism Improvements (Avoid Gaming/CGI Aesthetics)

The generated video currently looks like a video game cutscene (plastic textures, CGI lightning) rather than a realistic film clip. Below are the identified shortcomings and planned tasks to transition the pipeline output to photorealistic cinematic clips.

## Identified Lackings (Root Causes)

1. **Prompt Style Word Polluting:**
   - Prompts include words like `"render"`, `"cyberpunk"`, `"hyperdetailed"`, `"glow/glowing"`, which signal text-to-video models to output CGI, 3D game engines (Unreal Engine), or stylized art instead of real footage.

2. **Absence of Negative Prompts:**
   - The generator does not supply negative prompts to the Wan 2.2 model. Without negative prompts (e.g. `"cgi, 3d render, video game, drawing, painting, cartoon"`), diffusion models tend to fall back on smooth, game-like aesthetics.

3. **Low Base Resolution & Soft Textures:**
   - Generating at `832x480` and upscaling makes fine details (film grain, skin pores, metallic scratches) blurry, which mimics video game textures.
   - 20 denoising steps can leave reflections looking plasticky due to incomplete noise convergence.

## Action Plan for Next Session

### Prompting Upgrades
- [x] Refactor script prompts to forbid words like `"render"`, `"CGI"`, `"3D"`, `"artistic"`.
- [x] Inject cinematic photographic anchors: `"shot on 35mm film"`, `"cinematic movie scene"`, `"live-action film capture"`, `"realistic lighting"`, `"photograph"`, `"Panavision anamorphic lens"`.

### Generator Config Upgrades
- [x] **Support Negative Prompts:** Modify the local generator endpoint on the remote GPU server (`gpu_server.py`) and the pipeline generator caller to accept and use a default negative prompt targeting game graphics:
  ```
  negative_prompt: "cgi, 3d render, video game, anime, cartoon, sketch, painting, drawing, unreal engine, blender, smooth surfaces, low quality"
  ```
  - [x] Increase steps to `30` for final production runs to sharpen textures.
  - [x] Test higher base rendering resolutions (e.g., `1024x576` or `1280x720`) on the H200 SXM card.

---

# New Session: Cost-Efficient Cinematic Optimization

Optimize generation parameters to achieve rapid, low-cost generations while maintaining high-quality cinematic results.

- [x] Configure `settings.yaml` with the cost-efficient dual-quality parameters:
  - [x] Set resolution to `832x480`
  - [x] Set `anchor_steps` to `30`
  - [x] Set `subsequent_steps` to `12`
- [x] Run pytest to verify all test suites pass.

