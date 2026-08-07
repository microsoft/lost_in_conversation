def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("database"):
        from tasks.database import TaskDatabase
        return TaskDatabase(**kwargs)
    elif task_name == "code":
        from tasks.code import TaskCode
        return TaskCode(**kwargs)
    elif task_name == "translation":
        from tasks.translation import TaskTranslation
        return TaskTranslation(**kwargs)
    elif task_name == "summary":
        from tasks.summary import TaskSummary
        return TaskSummary(**kwargs)
    elif task_name == "data2text":
        from tasks.data2text import TaskData2Text
        return TaskData2Text(**kwargs)
    elif task_name == "math":
        from tasks.math import TaskMath
        return TaskMath(**kwargs)
    elif task_name.startswith("actions"):
        from tasks.actions import TaskActions
        return TaskActions(**kwargs)
    elif task_name == "swe":
        from tasks.swe import TaskSWE
        return TaskSWE(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")

if __name__ == "__main__":
    task = get_task("data2text")
    print(len(task.get_samples()))
