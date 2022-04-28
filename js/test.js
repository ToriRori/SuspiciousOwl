import * as so from "./main.js"

const imageFile = document.getElementById("imageFile")
const startBtn = document.getElementById("startBtn")
const canvasImage = document.createElement('canvas')
const ctxImage = canvasImage.getContext('2d')

imageFile.addEventListener('change', function(e) {
    const files = imageFile.files[0];
    if (files) {
        const fileReader = new FileReader();
        fileReader.onload = function(event){
            var img = new Image();
            img.onload = function(){
                canvasImage.width = img.width;
                canvasImage.height = img.height;
                ctxImage.drawImage(img,0,0);
            }
            img.src = event.target.result;
        }
        fileReader.readAsDataURL(e.target.files[0]);
    }
})

function isCanvasBlank(canvas) {
    return !canvas.getContext('2d')
        .getImageData(0, 0, canvas.width, canvas.height).data
        .some(channel => channel !== 0);
}

startBtn.addEventListener('click', async function() {
    if (isCanvasBlank(canvasImage)) {
        so.addChecker(console.log)
    } else {
        so.addChecker(console.log, canvasImage)
    }
})
