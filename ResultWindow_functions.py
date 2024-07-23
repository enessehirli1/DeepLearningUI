import torch
import psycopg2
from dotenv import load_dotenv
import os
from PyQt5.QtWidgets import QTreeWidgetItem, QApplication, QMessageBox, QFileDialog
load_dotenv('.gitignore/db_info.env')


def save_results_to_db(epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc,
                       test_precision, test_recall, test_f1):
    try:
        # PostgreSQL veritabanına bağlan
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        cur = conn.cursor()

        # Veritabanına verileri ekle
        insert_query = """
        INSERT INTO training_results (epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc, test_precision, test_recall, test_f1)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
        epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc, test_precision,
        test_recall, test_f1))

        # Değişiklikleri kaydet ve bağlantıyı kapat
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Veritabanına yazılırken hata oluştu: {e}")


class ResultWindowFunctions:
    def __init__(self, ui):
        self.ui = ui
        self.ui.pushButton_kaydet.clicked.connect(self.save_results)
        self.ui.pushButton_kaydetme.clicked.connect(self.close_window)

    def update_results(self, epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc,
                       test_precision, test_recall, test_f1):
        QTreeWidgetItem(self.ui.treeWidget_sonuclar, [
            str(epoch),
            f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}", f"{train_f1:.4f}",
            f"{test_loss:.4f}", f"{test_acc:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}", f"{test_f1:.4f}"
        ])
        self.ui.progressBar_sonuclar.setValue(epoch + 1)
        QApplication.processEvents()
        # Son epoch kontrolü
        if epoch == self.ui.progressBar_sonuclar.maximum() - 1:
            self.final_results = (
            epoch, train_loss, train_acc, train_precision, train_recall, train_f1, test_loss, test_acc, test_precision,
            test_recall, test_f1)
            self.model = self.ui.widget.window().model  # Modeli kaydetmek için ekledik

    def save_results(self):
        if hasattr(self, 'final_results') and hasattr(self, 'model'):
            save_results_to_db(*self.final_results)
            file_path, _ = QFileDialog.getSaveFileName(None, "Modeli Kaydet", "", "PyTorch Model Files (*.pth)")
            if file_path:
                torch.save(self.model.state_dict(), file_path)
                QMessageBox.information(self.ui.treeWidget_sonuclar, 'Başarılı',
                                        'Model Sonuçları ve Model Başarıyla Kaydedildi!.')
                print("Sonuçlar ve model başarıyla kaydedildi.")
            else:
                QMessageBox.warning(self.ui.treeWidget_sonuclar, 'Hata', 'Model kaydedilemedi.')
                print("Model kaydedilemedi.")
        else:
            print("Kaydedilecek sonuç veya model bulunamadı.")
            QMessageBox.warning(self.ui.treeWidget_sonuclar, 'Hata', 'Kaydedilecek sonuç veya model bulunamadı.')

    def close_window(self):
        self.ui.widget.window().close()
