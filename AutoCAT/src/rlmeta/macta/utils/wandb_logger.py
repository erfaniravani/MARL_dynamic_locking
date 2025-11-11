import os
import wandb
import torch

class WandbLogger:
    def __init__(self,
                 project,
                 config):
        wandb.init(project=project, config=config)
        self.dir = wandb.run.dir

    def get_dir(self):
        return self.dir

    def log_student(self, student_stats,teacher_stats):
        if type(student_stats) == dict:
            stats = {}
            for k,v in student_stats.items():
                v = stats_filter(v, prefix=k+"_student")
                stats.update(v)
            student_stats = stats
            stats = {}
            for k,v in teacher_stats.items():
                v = stats_filter(v, prefix=k+"_teacher")
                stats.update(v)
            teacher_stats = stats
        else:
            # decode train stats:
            student_stats = stats_filter(student_stats, prefix="student")
            teacher_stats = stats_filter(teacher_stats, prefix="teacher")

        student_stats.update(teacher_stats)
        stats = student_stats
        wandb.log(stats)


    def log(self,
            train_stats,
            eval_stats):
        
        if type(train_stats) == dict:
            stats = {}
            for k,v in train_stats.items():
                v = stats_filter(v, prefix=k+"_train")
                stats.update(v)
            train_stats = stats
            stats = {}
            for k,v in eval_stats.items():
                v = stats_filter(v, prefix=k+"_eval")
                stats.update(v)
            eval_stats = stats
        else:
            # decode train stats:
            train_stats = stats_filter(train_stats, prefix="train")
            eval_stats = stats_filter(eval_stats, prefix="eval")

        train_stats.update(eval_stats)
        stats = train_stats
        wandb.log(stats)

    def save(self, model_num, model, prefix=""):
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, prefix+"model-{}.th".format(model_num)))

def stats_filter(stats,
                prefix='train'):
    stats = stats.dict()
    res_dict = {}
    for k, v in stats.items():
        k = prefix+"_"+k
        res_dict[k] = v['mean']
    return res_dict
