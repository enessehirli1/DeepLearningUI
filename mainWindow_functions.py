from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from PyQt5.QtCore import QThread, pyqtSignal
import os
import torch
from torchvision import transforms
from models import createEffNetB7, createEffNetB2, createViTB16, createResNet50
from engine import train, create_dataloaders
from ResultWindow import Ui_Dialog as ResultsWindow
from ResultWindow_functions import ResultWindowFunctions

class TrainingThread(QThread):
    update_signal = pyqtSignal(int, float, float, float, float, float, float, float, float, float, float)

    def __init__(self, model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

    def run(self):
        train(
            model=self.model,
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            device=self.device,
            update_callback=self.update_callback
        )

    def update_callback(self, epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc, test_precision, test_recall, test_f1):
        self.update_signal.emit(epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc, test_precision, test_recall, test_f1)

class MainWindowFunctions:
    def __init__(self, ui):
        self.ui = ui
        self.train_folder = None
        self.test_folder = None
        self.num_classes = 0

    def select_train_folder(self):
        folder = QFileDialog.getExistingDirectory(None, "Train Klasör Seçiniz", "C:/Users/enes/Desktop")
        if folder:
            self.train_folder = folder
            self.ui.lineEdit_trainklasor.setText(folder)
            self.check_consistency()
        else:
            QMessageBox.information(None, "Train Klasörü", "Train klasörü seçimi iptal edildi")

    def select_test_folder(self):
        folder = QFileDialog.getExistingDirectory(None, "Test Klasör Seçiniz", "C:/Users/enes/Desktop")
        if folder:
            self.test_folder = folder
            self.ui.lineEdit_testklasor.setText(folder)
            self.check_consistency()
        else:
            QMessageBox.information(None, "Test Klasörü", "Test klasörü seçimi iptal edildi")

    def check_consistency(self):
        if self.train_folder and self.test_folder:
            train_classes = set(os.listdir(self.train_folder))
            test_classes = set(os.listdir(self.test_folder))

            # Sadece klasörleri kontrol et
            train_classes = {d for d in train_classes if os.path.isdir(os.path.join(self.train_folder, d))}
            test_classes = {d for d in test_classes if os.path.isdir(os.path.join(self.test_folder, d))}

            if train_classes != test_classes:
                QMessageBox.warning(None, "Hata", "Train ve Test klasörlerindeki sınıf isimleri aynı olmalıdır.")
                self.train_folder = None
                self.test_folder = None
                self.ui.lineEdit_trainklasor.clear()
                self.ui.lineEdit_testklasor.clear()
            else:
                self.num_classes = len(train_classes)
                QMessageBox.information(None, "Başarılı", "Train ve Test klasörleri başarıyla seçildi.")

    def start_training(self):
        # Kullanıcıdan parametreleri alın
        model_name = self.get_selected_model()
        optimizer_name = self.get_selected_optimizer()
        loss_fn_name = self.get_selected_loss_fn()
        epochs = int(self.ui.lineEdit_epochsayisi.text())
        train_folder = self.ui.lineEdit_trainklasor.text()
        test_folder = self.ui.lineEdit_testklasor.text()

        # Model, optimizer, loss function ve diğer parametreleri burada oluşturun
        model, optimizer, loss_fn = self.create_model_optimizer_loss_fn(model_name=model_name, optimizer_name=optimizer_name, loss_fn_name=loss_fn_name)

        # DataLoader'ları oluşturun
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_folder, test_folder, transform, batch_size=32)

        # Eğitim sonuçlarını gösterecek yeni pencereyi açın
        self.results_window = QDialog()
        self.results_ui = ResultsWindow()
        self.results_ui.setupUi(self.results_window)
        self.result_functions = ResultWindowFunctions(self.results_ui)
        self.results_ui.progressBar_sonuclar.setMaximum(epochs)
        self.results_window.show()

        # Modeli ResultWindowFunctions'a aktar
        self.results_ui.widget.window().model = model

        # Eğitim işlemi için thread oluşturun
        self.training_thread = TrainingThread(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device='cuda')
        self.training_thread.update_signal.connect(self.result_functions.update_results)
        self.training_thread.start()

    def get_selected_model(self):
        if self.ui.radioButton_effnetb7.isChecked():
            return "EffNetB7"
        elif self.ui.radioButton_resnet50.isChecked():
            return "ResNet50"
        elif self.ui.radioButton_vitb16.isChecked():
            return "ViTB16"
        elif self.ui.radioButton_effnetb2.isChecked():
            return "EffNetB2"

    def get_selected_optimizer(self):
        if self.ui.radioButton_Adam.isChecked():
            return "Adam"
        elif self.ui.radioButton_SGD.isChecked():
            return "SGD"

    def get_selected_loss_fn(self):
        if self.ui.radioButton_crossentropyloss.isChecked():
            return "CrossEntropyLoss"
        elif self.ui.radioButton_nlloss.isChecked():
            return "NLLoss"

    def create_model_optimizer_loss_fn(self, model_name, optimizer_name, loss_fn_name):
        # Model, optimizer ve loss function'ı oluşturma işlemleri
        if model_name == "EffNetB7":
            model, _ = createEffNetB7(num_classes=self.num_classes)
        elif model_name == "ResNet50":
            model, _ = createResNet50(num_classes=self.num_classes)
        elif model_name == "ViTB16":
            model, _ = createViTB16(num_classes=self.num_classes)
        elif model_name == "EffNetB2":
            model, _ = createEffNetB2(num_classes=self.num_classes)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters())
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        if loss_fn_name == "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn_name == "NLLoss":
            loss_fn = torch.nn.NLLLoss()

        return model, optimizer, loss_fn
