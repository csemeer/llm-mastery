"""
Training Manager

Manages model training and fine-tuning jobs.
"""

import threading
import time
from datetime import datetime
from typing import Dict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import db, TrainingJob


class TrainingManager:
    """Manages model training jobs"""

    def __init__(self, socketio):
        self.socketio = socketio
        self.running_jobs: Dict[int, threading.Thread] = {}
        self.job_stop_flags: Dict[int, threading.Event] = {}

    def start_training(self, job: TrainingJob):
        """Start training job"""
        if job.id in self.running_jobs and self.running_jobs[job.id].is_alive():
            raise ValueError("Job is already running")

        # Create stop flag
        self.job_stop_flags[job.id] = threading.Event()

        # Start training thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job.id,),
            daemon=True
        )
        thread.start()

        self.running_jobs[job.id] = thread

    def cancel_training(self, job_id: int):
        """Cancel training job"""
        job = TrainingJob.query.get(job_id)

        if not job:
            return

        if job_id in self.job_stop_flags:
            self.job_stop_flags[job_id].set()

        job.status = 'cancelled'
        db.session.commit()

        self.socketio.emit('training_cancelled', {
            'job_id': job_id
        })

    def _run_training(self, job_id: int):
        """Run training job"""
        print(f"Starting training job {job_id}")

        job = TrainingJob.query.get(job_id)
        stop_flag = self.job_stop_flags[job_id]

        try:
            job.status = 'running'
            job.started_at = datetime.utcnow()
            db.session.commit()

            config = json.loads(job.config) if job.config else {}
            total_epochs = config.get('epochs', 50)
            job.total_epochs = total_epochs
            db.session.commit()

            # Training loop
            for epoch in range(1, total_epochs + 1):
                if stop_flag.is_set():
                    break

                # Simulate training (in production, call actual training code)
                time.sleep(2)  # Simulate epoch time

                # Update progress
                job.current_epoch = epoch
                job.progress = (epoch / total_epochs) * 100
                job.loss = 0.5 - (epoch * 0.01)  # Simulated decreasing loss
                job.accuracy = 0.5 + (epoch * 0.01)  # Simulated increasing accuracy
                db.session.commit()

                # Emit progress update
                self.socketio.emit('training_progress', {
                    'job_id': job_id,
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'progress': job.progress,
                    'loss': job.loss,
                    'accuracy': job.accuracy
                })

            # Mark as completed if not cancelled
            if not stop_flag.is_set():
                job.status = 'completed'
                job.completed_at = datetime.utcnow()
                job.model_path = f'checkpoints/model_job_{job_id}.pt'
                db.session.commit()

                self.socketio.emit('training_completed', {
                    'job_id': job_id,
                    'model_path': job.model_path
                })

        except Exception as e:
            print(f"Error in training job {job_id}: {str(e)}")
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.session.commit()

            self.socketio.emit('training_failed', {
                'job_id': job_id,
                'error': str(e)
            })

        finally:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            if job_id in self.job_stop_flags:
                del self.job_stop_flags[job_id]

            print(f"Finished training job {job_id}")
