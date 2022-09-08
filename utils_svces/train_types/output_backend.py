import pathlib
import os
from datetime import datetime
import time
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from .train_loss import Log, LogType
import time

class OutputBackend:
    def __init__(self, base_model_dir, log_dir, type_description, print_output=True, batch_update_interval=10):
        self._create_model_dirs(base_model_dir, log_dir, type_description)
        self.writer = SummaryWriter(self.writer_dir)
        self.epoch_t_average = 0
        self.epoch_t_N = 0
        self.print_output = print_output
        self.batch_update_interval = batch_update_interval

    def close_backend(self):
        self.writer.close()
        self._finalize_model_dirs()

    def log_epoch_time(self, epoch_t, epoch, total_epochs):
        self.writer.add_scalar('EpochTime', epoch_t, epoch)
        self.epoch_t_average = self.epoch_t_average + (epoch_t - self.epoch_t_average) / min(self.epoch_t_N + 1, 5)
        self.epoch_t_N += 1

        if self.print_output:
            estimate_remaining_time = (total_epochs - epoch - 1) * self.epoch_t_average
            print(f'Avg. epoch time {self.epoch_t_average:.3f}m - Remaining time {estimate_remaining_time:.3f}m')

    def _write_logs_inner(self, losses_logs, epoch, train, category):
        if train:
            train_prefix = 'Train'
        else:
            train_prefix = 'Test'
        for log in losses_logs:
            tag = f'{train_prefix}/{category}/{log.name}'

            if log.type is LogType.SCALAR:
                self.writer.add_scalar(tag, log.value, epoch)
            elif log.type is LogType.HISTOGRAM:
                self.writer.add_histogram(tag, log.value, epoch)
            else:
                raise ValueError(f'{log.name} passed not supported type')

    def write_losses(self, losses, epoch, train):
        losses_logs_combined = []
        for loss in losses:
            if loss is not None:
                losses_logs_combined.extend(loss.get_logs())

        self._write_logs_inner(losses_logs_combined, epoch, train, 'Losses')

    def write_loggers(self, loggers, epoch, train):
        logges_logs_combined = []
        for loss in loggers:
            if loss is not None:
                logges_logs_combined.extend(loss.get_logs())

        self._write_logs_inner(logges_logs_combined, epoch, train, 'Statistics')

    #log all losses and loggers
    def write_epoch_summary(self, losses, loggers, epoch, train):
        self.write_losses(losses, epoch, train)
        self.write_loggers(loggers, epoch, train)

    #convenience function
    def end_epoch_write_summary(self, losses, loggers, epoch, train):
        self.write_epoch_summary(losses, loggers, epoch, train)
        self.end_epoch_log()

    #start epoch log
    def start_epoch_log(self, total_batches):
        self.pbar = tqdm(total=total_batches,  bar_format='{l_bar}{bar:5}{r_bar}{bar:-5b}')

    #end epoch log
    def end_epoch_log(self):
        self.pbar.update(self.pbar.total - self.pbar.n)
        self.pbar.close()

    def log_batch_summary(self, epoch, batch_idx, train, losses=None, loggers=None):
        if (batch_idx % self.batch_update_interval) == 0:
            if train:
                pre_string = f'Train epoch {epoch}'
            else:
                pre_string = f'Test epoch {epoch}'

            post_string = ''
            losses_loggers = []
            if losses is not None:
                losses_loggers.extend(losses)
            if loggers is not None:
                losses_loggers.extend(loggers)

            for loss in losses_loggers:
                if loss is not None:
                    loss_logs = loss.get_logs()
                    for log in loss_logs:
                        if log.type is LogType.SCALAR:
                            post_string_add = f'{log.name:} {log.value:.4f} '
                            post_string += post_string_add
                        else:
                            #ignore non scalar logs
                            pass

            self.pbar.set_postfix_str(post_string)
            self.pbar.set_description_str(pre_string)
            self.pbar.update(n= batch_idx - self.pbar.n)
            time.sleep(0.01)


    def _create_model_dirs(self, base_model_dir, log_dir, type_description):
        created = False
        while not created:
            try:
                date_stamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
                self.model_name = f'{type_description}_{date_stamp}'
                self.temp_dir = os.path.join(base_model_dir, f'_temp_{self.model_name}')
                self.main_dir = os.path.join(base_model_dir, self.model_name)
                self.writer_dir = os.path.join(log_dir, self.model_name)
                self.checkpoints_dir = os.path.join(self.temp_dir, 'checkpoints')
                pathlib.Path(self.temp_dir).mkdir(parents=True, exist_ok=False)
                pathlib.Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=False)
                pathlib.Path(self.writer_dir).mkdir(parents=True, exist_ok=False)
                created = True
            except FileExistsError:
                print(f'Warning: Directory {self.temp_dir} already exists')
                time.sleep(1)

        print(f'Model final dir: {self.main_dir} - temp dir {self.temp_dir}')

    def save_model_checkpoint(self, model_state_dict, epoch, optimizer_state_dict=None, avg=False):
        if avg:
            avg_postfix = '_avg'
        else:
            avg_postfix = ''

        if epoch in ['best', 'final', 'best_swa', 'final_swa']:
            checkpoint_file = os.path.join(self.temp_dir, f'{epoch}{avg_postfix}.pth')
            optimizer_file = os.path.join(self.temp_dir, f'{epoch}_optim.pth')
        else:
            checkpoint_file = os.path.join(self.checkpoints_dir, f'{epoch}{avg_postfix}.pth')
            optimizer_file = os.path.join(self.checkpoints_dir, f'{epoch}_optim.pth')

        torch.save(model_state_dict, checkpoint_file)

        if optimizer_state_dict is not None:
            torch.save(optimizer_state_dict, optimizer_file)

    def _finalize_model_dirs(self):
        os.rename(self.temp_dir, self.main_dir)

    @staticmethod
    def _create_dict_markdown_text(config_dict, text, indent_level=0):
        new_txt = text
        if config_dict is not None:

            if indent_level <= 1:
                heading_level = min(6, indent_level + 3 )
                heading_pre = heading_level * '#' + ''
                heading_post = ''
            else:
                heading_pre = '**'
                heading_post = ':**'

            for i, (key_i, item_i) in enumerate(config_dict.items()):
                if type(item_i) is dict:
                    new_txt += f'{heading_pre}{key_i}{heading_post}    \n'
                    new_txt = OutputBackend._create_dict_markdown_text(item_i, new_txt, indent_level + 1)
                else:
                    if isinstance(item_i, (bool, int, float, complex, str, list, tuple)):
                     new_txt += f'**{key_i}:** {item_i}    \n'
            new_txt += '\n\n'

            return new_txt

    @staticmethod
    def _save_dict_to_txt(config_dict, fileID, indent_level=0):
        if config_dict is not None:
            pre = indent_level * '\t'
            for i, (key_i, item_i) in enumerate(config_dict.items()):
                if type(item_i) is dict:
                    fileID.write( f'{pre}{key_i}\n\n' )
                    OutputBackend._save_dict_to_txt(item_i, fileID, indent_level + 1)
                else:
                    if isinstance(item_i, (bool, int, float, complex, str, list, tuple)):
                     fileID.write( f'{pre}{key_i}: {item_i}\n' )
            fileID.write( '\n' )



    def save_model_configs(self, configs):
        out_file = os.path.join(self.temp_dir, 'config.txt')
        with open(out_file, 'w') as fileID:
            OutputBackend._save_dict_to_txt(configs, fileID)

        markdown_text = OutputBackend._create_dict_markdown_text(configs, '')
        self.writer.add_text(f'Params/config', markdown_text)


