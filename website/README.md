# Admixture Identifiability — Results Website

Lightweight React + TypeScript + Vite site that presents the experimental
results from the Gaussian bump flow-matching pipeline.

## Development

```bash
cd website
npm install
npm run dev          # start dev server with hot reload
npm run build        # production build → dist/
npm run preview      # preview production build locally
```

## Deployment

The repo includes a `netlify.toml` at the root that tells Netlify to:

1. Use `website/` as the build base.
2. Run `npm run build`.
3. Publish the `dist/` directory.

## Adding new results

1. Copy the result PNGs into `public/results/<case_name>/`.
2. Update `src/App.tsx` to reference the new case.
3. Rebuild and push.
