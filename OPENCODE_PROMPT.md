Build the frontend for HalluciGuard's live Blackboard demo.

Read WORKFLOW.md first — it has the full request flow, the exact /analyze
and /health API contracts (request/response JSON shapes), and the frontend
requirements list. Build against that contract exactly; don't guess field
names.

Use the ui-ux-max skill for all visual and UX decisions — layout, type,
color, spacing, motion. No design direction from me beyond what's already
in WORKFLOW.md's "Frontend requirements" and "Explainability content"
sections.

Backend (blackboard_core.py, server.py, requirements.txt) is already built
and tested — don't modify it. Put your frontend in static/index.html
(server.py already serves that path). Same-origin, so call /analyze and
/health as relative paths, no CORS handling needed.

This demos live in front of a college evaluator in 2 days, so: handle the
loading state (real requests take several seconds), handle /analyze errors
visibly, and make the verdict states (SUPPORTED / CONTRADICTED /
INSUFFICIENT) unmistakable at a glance.
