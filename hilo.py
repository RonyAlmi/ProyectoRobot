from __future__ import print_function

import time
import threading
import cv2


class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # read () se bloquea hasta que haya un nuevo frame
        self.cond = threading.Condition()

        # Permite detectar el hilo satisfactoriamente
        self.running = False

        # Mantiene el frame mas nuevo
        self.frame = None

        # Pasa un número de secuencia para que read () NO se bloquee
        self.latestnum = 0

        # Variable usada para mostrar en pantalla
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            try:

                # Bloque para marco fresco
                (rv, img) = self.capture.read()
                #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                assert rv
                counter += 1

                # Publica el frame
                with self.cond:  # Bloquea la condición para realizar esta operación
                    self.frame = img if rv else None
                    self.latestnum = counter
                    self.cond.notify_all()

                if self.callback:
                    self.callback(img)
            except:
                pass

    def read(self, wait=True, seqnumber=None, timeout=None):
        # Sin argumentos (esperar = Verdadero), siempre se bloquea para un frame nuevo
        # Con wait = False devuelve el fotograma actual inmediatamente (sondeo)
        # Con un número de secuencia, se bloquea hasta que ese frame está disponible (o no hay espera)
        # Con argumento de tiempo de espera, puede devolver un frame anterior;
        # Incluso puede ser (0, Ninguno) si aún no se ha recibido nada

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return self.latestnum, self.frame

            return self.latestnum, self.frame

