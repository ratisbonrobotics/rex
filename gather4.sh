wget -q https://github.com/JoeyTeng/jaxrenderer/releases/download/v0.3.1/brax-env-params-positional.zip -O policies.zip
rm -rf params_* 2>&1 >/dev/null
unzip -q policies.zip