from clearml import PipelineController


def run_silent_script_pipeline():
    # 1. Initialize the Controller
    pipe = PipelineController(
        name="ImConvo_EndToEnd_Pipeline",
        project="ImConvo",
        version="1.0.0",
        add_pipeline_tags=True,
    )

    pipe.add_step(
        name="download_data",
        base_task_project="ImConvo",
        base_task_name="data_ingestion",
    )
    # 2. STEP 1: Preprocessing (The "Heaviest" node)
    # This will trigger scripts/preprocess.py
    pipe.add_step(
        name="preprocess_data",
        base_task_project="ImConvo",
        base_task_name="preprocess_data",
        parents=[
            "download_data",
        ],
        parameter_override={
            "Args/parent": "${download_data.task_id}"  # 1 single parent
        },
        # ClearML will look for the task that ran preprocess.py
        # and clone it with these new parameters
    )

    # 3. STEP 2: Training (Dependent on Preprocessing)
    pipe.add_step(
        name="train_model",
        parents=[
            "preprocess_data",
        ],
        base_task_project="ImConvo",
        base_task_name="LipReadingCTC_Training",
        parameter_override={
            "General/parents_ids": "${[preprocess_data.task_id]}",  # pass in as a list of parents
        },
    )

    # 4. STEP 3: Evaluation (The "Gatekeeper")
    # This triggers test.py
    pipe.add_step(
        name="evaluate_model",
        parents=["train_model"],
        base_task_project="ImConvo",
        base_task_name="Model_Evaluation_Task",
        parameter_override={
            "Args/train_task_id": "${train_model.task_id}",
            "Args/preprocess_task_id": "${preprocess_data.task_id}",
        },
    )

    # 5. Execute on the Queue (Your "Real Worker" PC will pick this up)
    pipe.start(queue="default")


if __name__ == "__main__":
    run_silent_script_pipeline()
