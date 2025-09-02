import qrcode
data="I LOVE YOU"
img=qrcode.make(data)
img.save("qrcodee.png")
print("SCAN THE QR CODE")