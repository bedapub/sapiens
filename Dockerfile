FROM python:3.9
LABEL maintainer="miles.henderson@roche.com"

ARG USER_UID=1000
ARG USERNAME=worker

# Change to non-root user
RUN  adduser -u ${USER_UID} ${USERNAME}
USER ${USER_UID}

# set workdir 
WORKDIR /app

# For every copy command we add the --chown option to make
# the non-root user to the owner of the files copied
COPY --chown=${USER_UID} ./requirements.txt /app/requirements.txt
COPY --chown=${USER_UID} ./sapiens /app/sapiens
COPY --chown=${USER_UID} ./pyproject.toml /app/pyproject.toml

# dependencies
RUN pip install -e .
RUN pip install -r /app/requirements.txt

