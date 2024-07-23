from login_python import Ui_Dialog
from PyQt5 import QtWidgets

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    loginWindow = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(loginWindow)
    loginWindow.show()
    sys.exit(app.exec_())
