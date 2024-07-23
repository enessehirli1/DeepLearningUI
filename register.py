import sys
import os
from PyQt5 import QtWidgets, uic
import psycopg2
from PyQt5.QtWidgets import QMessageBox
from dotenv import load_dotenv

load_dotenv('.gitignore/db_info.env')

class RegisterWindow(QtWidgets.QDialog):
    def __init__(self):
        super(RegisterWindow, self).__init__()
        uic.loadUi('register.ui', self)
        self.pushButton_kayitol_kayit.clicked.connect(self.register_user)

    def register_user(self):
        username = self.lineEdit_kullaniciadi_kayit.text()
        password = self.lineEdit_sifre_kayit1.text()
        confirm_password = self.lineEdit_sifre_kayit_2.text()

        if not username or not password or not confirm_password:
            QMessageBox.warning(self, 'Hata', 'Tüm alanlar doldurulmalıdır.')
            return

        if password != confirm_password:
            QMessageBox.warning(self, 'Hata', 'Şifreler uyuşmuyor.')
            return

        if self.save_user(username, password):
            QMessageBox.information(self, 'Başarılı', 'Kayıt başarıyla tamamlandı.')
            self.close()
        else:
            QMessageBox.warning(self, 'Hata', 'Kullanıcı adı zaten mevcut.')

    def save_user(self, username, password):
        try:
            connection = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
            )
            cursor = connection.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            connection.commit()
            connection.close()
            return True
        except psycopg2.IntegrityError:
            return False
        except Exception as e:
            print(e)
            return False

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    register_window = RegisterWindow()
    register_window.show()
    sys.exit(app.exec_())
