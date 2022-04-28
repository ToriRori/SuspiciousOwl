import {faceapi} from './faceapi2.js'
import './tf.js'

var canvasFace
var webcamFace

const webcam = document.createElement("video")
webcam.autoplay = true

const model_url = 'models'
const tfPath = "models/best_web_model/model.json"

var image, model, input


window.addEventListener('load', async function() {
    await loadModels()
})

async function loadModels() {
    model = await tf.loadGraphModel(tfPath)

    await faceapi.loadTinyFaceDetectorModel(model_url)
    await faceapi.loadFaceRecognitionModel(model_url)
    await faceapi.loadFaceExpressionModel(model_url)
    await faceapi.loadFaceLandmarkTinyModel(model_url)
    console.log("Faceapi and yolo models are loaded")
}

export async function addChecker(violationCallback, expectedCanvas, smartphonesCheck=true, angleCheck=true, xAngleThreshold=10, yAngleThreshold=30, similarityCheck=true, similarity_threshold=0.4, tabCheck=true, fullsizeCheck=true, focusCheck=true) {
    addWebcamChecker(violationCallback, expectedCanvas, smartphonesCheck, angleCheck, xAngleThreshold, yAngleThreshold, similarityCheck, similarity_threshold)
    addWindowActivityChecker(violationCallback, tabCheck, fullsizeCheck, focusCheck)
}

export async function addWebcamChecker(violationCallback, expectedCanvas, smartphonesCheck=true, angleCheck=true, xAngleThreshold=10, yAngleThreshold=30, similarityCheck=true, similarity_threshold=0.4) {
    await runVideo()
    setInterval(checkWebcam, 1000, violationCallback, expectedCanvas, webcam, smartphonesCheck, angleCheck, xAngleThreshold, yAngleThreshold, similarityCheck, similarity_threshold)
}

async function runVideo() {
    const constraints = {
        video: 1
    }
    webcam.srcObject = await navigator.mediaDevices.getUserMedia(constraints)
}

async function checkWebcam(violationCallback, expectedCanvas, actualCanvas=webcam, smartphonesCheck=true, angleCheck=true, xAngleThreshold=10, yAngleThreshold=30, similarityCheck=true, similarity_threshold=0.4) {
    var result = {
        people_num: 0,
        similarity_value: 0,
        x_angle: 0,
        y_angle: 0,
        phones_num: 0,
    }

    webcamFace = await DetectFaces(actualCanvas)

    if (smartphonesCheck) {
        result['phones_num'] = await findSmartphone(actualCanvas, violationCallback)
    }

    if (angleCheck && webcamFace[0]) {
        var angle = checkAngle(violationCallback, xAngleThreshold, yAngleThreshold)
        result.y_angle = angle[0]
        result.x_angle = angle[1]
    }

    if (expectedCanvas) {
        canvasFace = await DetectFaces(expectedCanvas)
    }

    result.people_num = webcamFace.length

    if (webcamFace.length > 1) {
        violationCallback("More then one person")
    } else if (canvasFace && similarityCheck && webcamFace[0] && canvasFace[0]) {
        result['similarity_value'] = checkSimilarity(canvasFace, webcamFace, similarity_threshold, violationCallback)
    } else if (!webcamFace[0]) {
        violationCallback("face not found")
    }

    return result
}

async function findSmartphone(inputCanvas, violationCallback, threshold=0.7) {
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
            var x1, y1, x2, y2, width, height
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
                violationCallback("Found smartphone")
            }
        }

        return result
    })
}

function checkAngle(violationCallback, xAngleThreshold=10, yAngleThreshold=30) {
    var angle = getFaceAngle(webcamFace[0].landmarks.positions)
    if (Math.abs(angle[0]) > yAngleThreshold)
        violationCallback("Y angle is broken")
    if (Math.abs(angle[1]) > xAngleThreshold)
        violationCallback("X angle is broken")
    return angle
}

function getFaceAngle(faceLandmarks) {
    let x_nose = faceLandmarks[33].x;
    let y_nose = faceLandmarks[33].y;
    let x_left = faceLandmarks[2].x;
    let y_left = faceLandmarks[2].y;
    let x_right = faceLandmarks[14].x;
    let y_right = faceLandmarks[14].y;
    let x_reye = faceLandmarks[45].x;
    let y_reye = faceLandmarks[45].y;
    let x_leye = faceLandmarks[36].x;
    let y_leye = faceLandmarks[36].y;
    let x_rmouse = faceLandmarks[54].x;
    let y_rmouse = faceLandmarks[54].y;
    let x_lmouse = faceLandmarks[48].x;
    let y_lmouse = faceLandmarks[48].y;
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

async function checkSimilarity(expectedFace, actualFace, threshold=0.4, violationCallback) {
    var dist = await faceapi.euclideanDistance(webcamFace[0].descriptor, canvasFace[0].descriptor)
    if (dist > 1 - threshold) {
        violationCallback("Actual person is not similar to expected")
    }

    return 1 - dist
}

async function DetectFaces(canvas) {
    return faceapi
        .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks(true)
        .withFaceDescriptors();
}

export function addWindowActivityChecker(violationCallback, tabCheck=true, fullsizeCheck=true, focusCheck=true) {
    if (focusCheck) {
        document.addEventListener('visibilitychange', function () {
            checkFocus(violationCallback)
        })
    }

    if (fullsizeCheck) {
        window.addEventListener('resize', function () {
            checkFullwindow(violationCallback)
        })
    }

    if (tabCheck) {
        window.addEventListener('focus', function () {
        })

        window.addEventListener('blur', function () {
            violationCallback("Tab was changed")
        })
    }
}

function checkFocus(violationCallback) {
    if (document.hidden) {
        violationCallback("Focus moved from window")
    }
}

function checkFullwindow(violationCallback) {
    if ((screen.width != window.innerWidth) || (window.screenX != 0) || (window.screenY != 0)) {
        violationCallback("Exit from fullscreen")
    }
}