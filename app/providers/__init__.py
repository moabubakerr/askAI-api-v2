"""Answer sources that are not SCAI's own pipeline.

Everything in here talks to a third party over the public internet, which the
rest of this codebase deliberately does not do (see app/core/config.py). Each
module is off unless its credentials are configured, so an on-prem deployment
that sets nothing keeps the original guarantee.
"""
