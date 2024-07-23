import sys
from PyQt5 import QtWidgets
import psycopg2
from PyQt5.QtWidgets import QMessageBox

main_window = None  # Global main window reference

def login_user(dialog):
    username = dialog.lineEdit_kullaniciadigiris.text()
    password = dialog.lineEdit_sifregiris.text()

    # Giriş alanları boşsa uyarı göster
    if not username or not password:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Hata')
        msg.setText('Kullanıcı adı ve şifre boş olamaz.')
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return

    try:
        # Veritabanı bağlantısı kurma
        connection = psycopg2.connect(
            database="dluiDB",
            user="postgres",
            password="root",
            host="localhost",
            port="5432"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()
        connection.close()

        if result:
            open_main_window(dialog)  # Giriş başarılı olduğunda ana pencereye geçiş yap
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle('Hata')
            msg.setText('Kullanıcı adı veya şifre hatalı.')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle('Hata')
        msg.setText(f'Bir hata oluştu: {str(e)}')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def open_main_window(dialog):
    global main_window
    from mainWindow import Ui_MainWindow
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    main_window.show()
    dialog.Dialog.close()  # Dialog'u kapatmak için
