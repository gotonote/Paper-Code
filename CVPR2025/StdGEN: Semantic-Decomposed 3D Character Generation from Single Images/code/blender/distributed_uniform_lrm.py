import json
import multiprocessing
import subprocess
import time
from dataclasses import dataclass
import os
import tyro
import concurrent.futures
@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""
    num_gpus: int = 8
    """number of gpus to use. -1 means all available gpus"""
    input_dir: str
    save_dir: str
    engine: str = "BLENDER_EEVEE"


def check_already_rendered(save_path):
    if not os.path.exists(os.path.join(save_path, '02419_semantic.png')):
        return False
    return True

def process_file(file):
    if not check_already_rendered(file[1]):
        return file
    return None

def worker(queue, count, gpu):
    while True:
        try:
            item = queue.get()
            if item is None:
                queue.task_done()
                break
            data_path, save_path, engine, log_name = item
            print(f"Processing: {data_path} on GPU {gpu}")
            start = time.time()
            if check_already_rendered(save_path):
                queue.task_done()
                print('========', item, 'rendered', '========')
                continue
            else:
                os.makedirs(save_path, exist_ok=True)
                command = (f"export DISPLAY=:0.{gpu} &&"
                            f" CUDA_VISIBLE_DEVICES={gpu} "
                            f" blender -b -P blender_lrm_script.py --"
                            f" --object_path {data_path} --output_dir {save_path} --engine {engine}")

                try:
                    subprocess.run(command, shell=True, timeout=3600, check=True)
                    count.value += 1
                    end = time.time()
                    with open(log_name, 'a') as f:
                        f.write(f'{end - start}\n')
                except subprocess.CalledProcessError as e:
                    print(f"Subprocess error processing {item}: {e}")
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired processing {item}: {e}")
                except Exception as e:
                    print(f"Error processing {item}: {e}")
                finally:
                    queue.task_done()

        except Exception as e:
            print(f"Error processing {item}: {e}")
            queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    log_name = f'time_log_{args.workers_per_gpu}_{args.num_gpus}_{args.engine}.txt'

    if args.num_gpus == -1:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        args.num_gpus = output.count('GPU')

    files = []

    for group in [ str(i) for i in range(10) ]:
        for folder in os.listdir(f'{args.input_dir}/{group}'):
            filename = f'{args.input_dir}/{group}/{folder}/{folder}.vrm'
            outputdir = f'{args.save_dir}/{group}/{folder}'
            files.append([filename, outputdir])

    # sorted the files
    files = sorted(files, key=lambda x: x[0])

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the process_file function to the files
        results = list(executor.map(process_file, files))

    # Filter out None values from the results
    unprocess_files = [file for file in results if file is not None]

    # Print the number of unprocessed files and the split ID
    print(f'Unprocessed files: {len(unprocess_files)}')

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

    for file in unprocess_files:
        queue.put((file[0], file[1], args.engine, log_name))

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu * 10):
        queue.put(None)
    # Wait for all tasks to be completed
    queue.join()
    end = time.time()
