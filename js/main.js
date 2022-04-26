import {faceapi} from './faceapi2.js'
import './tf.js'

var canvasFace
var webcamFace
var isFullWindow = true
var isActiveStatus = true
var onTabStatus = true

const webcam = document.createElement("video")
webcam.autoplay = true

const model_url = 'models'
const tfPath = "models/best_web_model/model.json"

var x1 = 0,
    x2 = 0,
    y1 = 0,
    y2 = 0,
    width = 0,
    height = 0
var image, model, input


function getAngle(pos) {
    let x_nose = pos[33].x;
    let y_nose = pos[33].y;
    let x_left = pos[2].x;
    let y_left = pos[2].y;
    let x_right = pos[14].x;
    let y_right = pos[14].y;
    let x_reye = pos[45].x;
    let y_reye = pos[45].y;
    let x_leye = pos[36].x;
    let y_leye = pos[36].y;
    let x_rmouse = pos[54].x;
    let y_rmouse = pos[54].y;
    let x_lmouse = pos[48].x;
    let y_lmouse = pos[48].y;
    let x_mid_eye = (x_reye + x_leye) / 2;
    let y_mid_eye = (y_reye + y_leye) / 2;
    let x_mid_mouse = (x_rmouse + x_lmouse) / 2;
    let y_mid_mouse = (y_rmouse + y_lmouse) / 2;
    let x_eye_mouse_mid = (x_mid_eye + x_mid_mouse) / 2;
    let y_eye_mouse_mid = (y_mid_eye + y_mid_mouse) / 2;
    let mid_eye_mouse_dist = Math.sqrt((x_mid_mouse - x_eye_mouse_mid) ** 2 + (y_mid_mouse - y_eye_mouse_mid) ** 2);
    let eye_nose_dist = Math.sqrt((x_mid_eye - x_nose) ** 2 + (y_mid_eye - y_nose) ** 2);
    let mouse_nose_dist = Math.sqrt((x_mid_mouse - x_nose) ** 2 + (y_mid_mouse - y_nose) ** 2);
    
    let nose_mouse_dist = Math.sqrt((x_eye_mouse_mid - x_nose) ** 2 + (y_eye_mouse_mid - y_nose) ** 2);
    if (eye_nose_dist > mouse_nose_dist) {
        nose_mouse_dist *= -1
    }

    let angle_yaxis = (nose_mouse_dist / mid_eye_mouse_dist) * 90;

    let length2 = Math.sqrt((x_left - x_right) ** 2 + (y_left - y_right) ** 2);

    let x_left_right_mid = (x_right + x_left) / 2;
    let y_left_right_mid = (y_right + y_left) / 2;

    let mid_nose_dist = Math.sqrt((x_left_right_mid - x_nose) ** 2 + (y_left_right_mid - y_nose) ** 2);
    let right_nose_dist = Math.sqrt((x_right - x_nose) ** 2 + (y_right - y_nose) ** 2);
    let left_nose_dist = Math.sqrt((x_left - x_nose) ** 2 + (y_left - y_nose) ** 2);
    
    if (left_nose_dist < right_nose_dist) {
        mid_nose_dist *= -1
    }

    let angle_xaxis = mid_nose_dist / length2 / 2 * 90;
    return [angle_yaxis, angle_xaxis]
}

async function loadModels() {
    await faceapi.loadTinyFaceDetectorModel(model_url)
    await faceapi.loadFaceRecognitionModel(model_url)
    await faceapi.loadFaceExpressionModel(model_url)
    await faceapi.loadFaceLandmarkTinyModel(model_url)
    console.log("Models are loaded")
}

async function runVideo() {
    const constraints = {
        video: 1
    }
    webcam.srcObject = await navigator.mediaDevices.getUserMedia(constraints)
}

async function getDistance(actualFace, expectedFace, similarityThreshold=0.4, xAngleThreshold=30, yAngleThreshold=10) { // args: canvases
    var result = {
        people_num: 0,
        same_person: false,
        similarity_value: 0,
        x_angle: 0,
        y_angle: 0,
        x_angle_broken: false,
        y_angle_broken: false
    }
    webcamFace = await faceapi
        .detectAllFaces(actualFace, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks(true)
        .withFaceDescriptors()
    canvasFace = await faceapi
        .detectAllFaces(expectedFace, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks(true)
        .withFaceDescriptors()

    result.people_num = webcamFace.length
    if (webcamFace[0] && canvasFace[0]) {
        var angle = getAngle(webcamFace[0].landmarks.positions)
        result.y_angle = angle[0]
        result.x_angle = angle[1]
        if (Math.abs(angle[0]) > yAngleThreshold)
            result.y_angle_broken = true
        if (Math.abs(angle[1]) > xAngleThreshold)
            result.x_angle_broken = true
        var dist = await faceapi.euclideanDistance(webcamFace[0].descriptor, canvasFace[0].descriptor)
        result.similarity_value = 1 - dist
        if (dist < 1 - similarityThreshold) {
            result.same_person = true
        }
    }

    return result
}

async function findSmartphone(inputCanvas, threshold=0.7) {
    var result = 0;
    image = tf.browser.fromPixels(inputCanvas)
    input = tf.tidy(() => {
        return tf.image
            .resizeBilinear(image, [640, 640])
            .div(255.0).
        expandDims(0);
    })
    await model.executeAsync(input).then(res => {
        const [boxes, scores, classes, valid_detections] = res
        const boxes_data = boxes.dataSync()
        const scores_data = scores.dataSync()
        const valid_detections_data = valid_detections.dataSync()[0]
        tf.dispose(res)

        for (var i = 0; i < valid_detections_data; ++i) {
            [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
            x1 *= 640
            x2 *= 640
            y1 *= 480
            y2 *= 480
            width = x2 - x1;
            height = y2 - y1;
            const score = scores_data[i].toFixed(3);
            if (score >= threshold) {
                result += 1
            }
        }

        return result
    })
}

export async function getStats(expectedCanvas, actualCanvas=webcam) {
    var smartphones = await findSmartphone(actualCanvas)
    var result = await getDistance(actualCanvas, expectedCanvas)
    result['phones_num'] = smartphones
    result['is_full_window'] = isFullWindow
    result['is_active_status'] = isActiveStatus
    result['on_tab_status'] = onTabStatus
    console.log(result)
    return result
}

export async function getStatsThread(expectedCanvas) {
    await runVideo()
    setInterval(getStats, 1000, expectedCanvas, webcam)
}

window.addEventListener('load', async function() {
    model = await tf.loadGraphModel(tfPath)
    await loadModels()
})

window.addEventListener('focus', function() {
    isActiveStatus = true
})

window.addEventListener('blur', function() {
    isActiveStatus = false
})

function checkOnTab() {
    onTabStatus = !document.hidden
}

function checkFullwindow() {
    isFullWindow = (screen.width == window.innerWidth) && (window.screenX == 0) && (window.screenY == 0)
}

document.addEventListener('visibilitychange', checkOnTab)
window.addEventListener('resize', checkFullwindow)