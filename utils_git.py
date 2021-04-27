import subprocess

def get_tag():
    """
    Check if there is a tag, but if there isn't a tags, then return None
    :return: str of the tag, or none
    """
    try:
        return subprocess.check_output(["git", "describe", '--tags']).strip().decode("ascii")
    except subprocess.CalledProcessError:
        return None

def get_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode("ascii")

def get_version():
    return {
            'tag': get_tag(),
            'commit': get_commit()
            }

if __name__ == '__main__':
    # in case there were any visits stuck in processing if something crashed
    # reset them here when turning on the API

    print(get_tag())
    print(get_commit())
    print(get_version())
