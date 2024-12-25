ARG python_version
FROM python:$python_version

LABEL maintainer="None"

# PYTHONUNBUFFERED keeps Python from buffering the standard
# output stream, which means that logs can be delivered to the user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are unnecessary in this case.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ARG requirements_file_path
ARG module_path
ARG target_dir_name

COPY ${requirements_file_path} /opt/program/easy_sm-requirements.txt
WORKDIR /opt/program/${target_dir_name}

# Here we get all python packages.
RUN pip install flask
RUN pip install -r ../easy_sm-requirements.txt && rm -rf /root/.cache

COPY ${module_path} /opt/program/${target_dir_name}

# The entrypoint is written specifically in this way because it needs to accept arguments at runtime
ENTRYPOINT ["easy_sm_base/executor.sh"]
