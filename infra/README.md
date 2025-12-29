# `infra/`

This folder is for infrastructure templates: everything needed to provision and configure the resources that run your ML workloads.

Terraform is a common default, but this template is tool-agnostic—use whatever fits your environment.

## What belongs here

- Modules/templates
- Environment definitions (dev/stage/prod)
- Deployment configuration (variables, parameters, values files)
- Documentation on how to provision, update, and destroy environments

## What should NOT live here

- Secrets (keys/tokens/passwords). Use environment variables or a secrets manager.
- Large artifacts or datasets (those belong under `data/`).
- Generated state files committed to source control unless you intentionally manage state that way.

## Suggested layout (example)

```text
infra/
  terraform/
    modules/
    envs/
      dev/
      prod/
  docs/
```

## Conventions (recommended)

- Separate environments (dev/stage/prod) cleanly.
- Prefer small reusable modules over one giant template.
- Keep a clear resource naming scheme.
- Document required permissions and prerequisites.
- Ensure state is stored safely (remote state and locking when applicable).

## How This Fits

- Supports operational execution from [`entrypoints/`](../entrypoints/) (scheduled jobs, batch runs, services)
- Often uses values/settings aligned with [`config/`](../config/) (but configs remain application-level)
- May provision storage/compute used by artifacts in [`data/`](../data/)

## Notes

- If you add CI/CD later, `infra/` is where provisioning and deployment steps usually live.
