import React, { Component, Fragment } from 'react';
import {
  Alert, Button, Collapse, Container, Form, Spinner, ListGroup, Tabs, Tab
} from 'react-bootstrap';
import { FaCamera, FaChevronDown, FaChevronRight } from 'react-icons/fa';
import { openDB } from 'idb';
import Cropper  from 'react-cropper';
import * as tf from '@tensorflow/tfjs';
import LoadButton from '../components/LoadButton';
import { MODEL_CLASSES } from '../model/classes';
import config from '../config';
import './Classify.css';
import 'cropperjs/dist/cropper.css';
import * as posenet from '@tensorflow-models/posenet';
import { abs, time } from '@tensorflow/tfjs';


const MODEL_PATH = '/model/model.json';
const CANVAS_SIZE = 224;
const TOPK_PREDICTIONS = 5;
const CONFIDENCE_THRESHOLD = 0.7
const ANGLE_THRESHOLD = 80
const STRAIGHTEN_THRESHOLD = 150

const INDEXEDDB_DB = 'tensorflowjs';
const INDEXEDDB_STORE = 'model_info_store';
const INDEXEDDB_KEY = 'web-model';

/**
 * Class to handle the rendering of the Classify page.
 * @extends React.Component
 */
export default class Classify extends Component {

  constructor(props) {
    super(props);

    this.webcam = null;
    this.model = null;
    this.modelLastUpdated = null;

    this.state = {
      modelLoaded: false,
      filename: '',
      isModelLoading: false,
      isClassifying: false,
      predictions: [],
      photoSettingsOpen: true,
      modelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showModelUpdateSuccess: false,
      isDownloadingModel: false
    };
    this.pushup_allowed = false;
    this.pushup_count = 0;
  }

  async componentDidMount() {
    if (('indexedDB' in window)) {
      try {
        this.model = await posenet.load({
          architecture: 'ResNet50',
          outputStride: 32,
          inputResolution: { width: 257, height: 200  },
          quantBytes: 2,
          modelUrl: 'indexeddb://' + INDEXEDDB_KEY
        })
        console.log('here')
        
        // tf.loadLayersModel('indexeddb://' + INDEXEDDB_KEY);

        // Safe to assume tensorflowjs database and related object store exists.
        // Get the date when the model was saved.
        try {
          const db = await openDB(INDEXEDDB_DB, 1, );
          const item = await db.transaction(INDEXEDDB_STORE)
                               .objectStore(INDEXEDDB_STORE)
                               .get(INDEXEDDB_KEY);
          const dateSaved = new Date(item.modelArtifactsInfo.dateSaved);
          await this.getModelInfo();
          if (!this.modelLastUpdated  || dateSaved >= new Date(this.modelLastUpdated).getTime()) {
            console.log('Using saved model');
          }
          else {
            this.setState({
              modelUpdateAvailable: true,
              showModelUpdateAlert: true,
            });
          }

        }
        catch (error) {
          console.warn(error);
          console.warn('Could not retrieve when model was saved.');
        }

      }
      // If error here, assume that the object store doesn't exist and the model currently isn't
      // saved in IndexedDB.
      catch (error) {
        console.log('please look at line 101')
        console.log('Not found in IndexedDB. Loading and saving...');
        console.log(error);
        this.model = await posenet.load({
          architecture: 'ResNet50',
          outputStride: 32,
          inputResolution: { width: 257, height: 200  },
          quantBytes: 2,
          // modelUrl: MODEL_PATH
        });
        // need to somehow save this to the db
        // await this.model.save('indexeddb://' + INDEXEDDB_KEY);
      }
    }
    // If no IndexedDB, then just download like normal.
    else {
      console.warn('IndexedDB not supported.');
      this.model = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 257, height: 200  },
        quantBytes: 2
      });
    }

    this.setState({ modelLoaded: true });
    this.initWebcam();

  
  }

  async componentWillUnmount() {
    if (this.webcam) {
      this.webcam.stop();
    }

    // Attempt to dispose of the model.
    try {
      this.model.dispose();
    }
    catch (e) {
      // Assume model is not loaded or already disposed.
    }
  }

  initWebcam = async () => {
    try {
      this.webcam = await tf.data.webcam(
        this.refs.webcam,
        {resizeWidth: CANVAS_SIZE, resizeHeight: CANVAS_SIZE, facingMode: 'environment'}
      );
    }
    catch (e) {
      this.refs.noWebcam.style.display = 'block';
    }
  }

  startWebcam = async () => {
    if (this.webcam) {
      this.webcam.start();
    }
  }

  stopWebcam = async () => {
    if (this.webcam) {
      this.webcam.stop();
    }
  }

  getModelInfo = async () => {
    await fetch(`${config.API_ENDPOINT}/model_info`, {
      method: 'GET',
    })
    .then(async (response) => {
      await response.json().then((data) => {
        this.modelLastUpdated = data.last_updated;
      })
      .catch((err) => {
        console.log('Unable to get parse model info.');
      });
    })
    .catch((err) => {
      console.log('Unable to get model info');
    });
  }

  updateModel = async () => {
    // Get the latest model from the server and refresh the one saved in IndexedDB.
    console.log('Updating the model: ' + INDEXEDDB_KEY);
    this.setState({ isDownloadingModel: true });
    this.model = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 640, height: 480 },
      multiplier: 0.50
    });
    await this.model.save('indexeddb://' + INDEXEDDB_KEY);
    this.setState({
      isDownloadingModel: false,
      modelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showModelUpdateSuccess: true
    });
  }


  classifyWebcamImage = async () => {
    this.setState({ isClassifying: true });
    this.pushup_count = 0;

    // warmup model to reduce wait time, gave 3 second buffer
    const imageCapture = await this.webcam.capture();
    const pose =  this.model.estimateSinglePose(imageCapture, 0.50, false, 16);


    document.getElementById('timer').hidden = false
    document.getElementById('timer').innerHTML = 'Starting in 3...'
    await new Promise(resolve => setTimeout(resolve, 1000))
    document.getElementById('timer').innerHTML = 'Starting in 2...'
    await new Promise(resolve => setTimeout(resolve, 1000))
    document.getElementById('timer').innerHTML = 'Starting in 1...'
    await new Promise(resolve => setTimeout(resolve, 1000))

    var start_time = Date.now()
    document.getElementById('pushup_count').hidden = false
    var count = 0
    while (Date.now() - start_time < 60000)
    {
      count += 1
      console.log(count)
      const imageCapture = await this.webcam.capture();
      const pose = await this.model.estimateSinglePose(imageCapture, 0.50, false, 16);
      if (await this.valid_pushup (pose, 5, 7, 9) || await this.valid_pushup (pose, 6, 8, 10)  )
      {
        if (this.pushup_allowed)
          {
            this.pushup_count +=1;
            this.pushup_allowed = false;
          }
      }
      if (Math.floor((60000 - (Date.now() - start_time))/1000) >= 0){
        document.getElementById('timer').innerHTML = Math.floor((60000 - (Date.now() - start_time))/1000) + ' seconds remaining'
      }
      else{
        document.getElementById('timer').innerHTML = '0 seconds remaining'
      }
      document.getElementById('pushup_count').innerHTML = this.pushup_count + ' Pushup(s) counted'
      if (this.pushup_allowed == false){
        document.getElementById('straighten_arm').hidden = false
      }
      else {
        document.getElementById('straighten_arm').hidden = true
      }
    }

    this.setState({
      isClassifying: false,
    });
    document.getElementById('pushup_count').innerHTML = this.pushup_count + ' Pushup(s) counted, click to restart'
  }


  valid_pushup = async (pose, shoulder_index, elbow_index, wrist_index) =>{
    var valid = false
    if (pose.keypoints[shoulder_index].score > CONFIDENCE_THRESHOLD && pose.keypoints[elbow_index].score > CONFIDENCE_THRESHOLD && pose.keypoints[wrist_index].score > CONFIDENCE_THRESHOLD) {
      {
          var forearm_length = Math.sqrt(Math.abs(pose.keypoints[shoulder_index].position.y - pose.keypoints[elbow_index].position.y) ** 2 + Math.abs(
            pose.keypoints[shoulder_index].position.x - pose.keypoints[elbow_index].position.x) ** 2);
          var bicep_length = Math.sqrt(Math.abs(pose.keypoints[elbow_index].position.y - pose.keypoints[wrist_index].position.y) ** 2 + Math.abs(
            pose.keypoints[elbow_index].position.x - pose.keypoints[wrist_index].position.x) ** 2);
          var exterior_length = Math.sqrt(Math.abs(pose.keypoints[shoulder_index].position.y - pose.keypoints[wrist_index].position.y) ** 2 + Math.abs(
            pose.keypoints[shoulder_index].position.x - pose.keypoints[wrist_index].position.x) ** 2);
          var angle = Math.acos((forearm_length ** 2 + bicep_length ** 2 - exterior_length ** 2) / (
            2 * forearm_length * bicep_length));
          angle = angle * (180 / Math.PI)    
          console.log(angle)
          if (angle < ANGLE_THRESHOLD && pose.keypoints[shoulder_index].position.y < pose.keypoints[wrist_index].position.y && pose.keypoints[elbow_index].position.y < pose.keypoints[wrist_index].position.y)
          {
            valid = true
          }
          console.log(angle, STRAIGHTEN_THRESHOLD)
          if (angle > STRAIGHTEN_THRESHOLD)
          {
            this.pushup_allowed = true
          }
        }
    }
    console.log(valid, this.pushup_allowed)
    return valid
  }

  handlePanelClick = event => {
    this.setState({ photoSettingsOpen: !this.state.photoSettingsOpen });
  }

  handleFileChange = event => {
    if (event.target.files && event.target.files.length > 0) {
      this.setState({
        file: URL.createObjectURL(event.target.files[0]),
        filename: event.target.files[0].name
      });
    }
  }

  handleTabSelect = activeKey => {
    switch(activeKey) {
      case 'camera':
        this.startWebcam();
        break;
      default:
    }
  }

  render() {
    return (
      <div className="Classify container">

      { !this.state.modelLoaded &&
        <Fragment>
          <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
          </Spinner>
          {' '}<span className="loading-model-text">Loading Model</span>
        </Fragment>
      }

      { this.state.modelLoaded &&
        <Fragment>
        <Button
          onClick={this.handlePanelClick}
          className="classify-panel-header"
          aria-controls="photo-selection-pane"
          aria-expanded={this.state.photoSettingsOpen}
          >
          Pushup Counter
            <span className='panel-arrow'>
            { this.state.photoSettingsOpen
              ? <FaChevronDown />
              : <FaChevronRight />
            }
            </span>
          </Button>
          <Collapse in={this.state.photoSettingsOpen}>
            <div id="photo-selection-pane">
            { this.state.modelUpdateAvailable && this.state.showModelUpdateAlert &&
                <Container>
                  <Alert
                    variant="info"
                    show={this.state.modelUpdateAvailable && this.state.showModelUpdateAlert}
                    onClose={() => this.setState({ showModelUpdateAlert: false})}
                    dismissible>
                      An update for the <strong>{this.state.modelType}</strong> model is available.
                      <div className="d-flex justify-content-center pt-1">
                        {!this.state.isDownloadingModel &&
                          <Button onClick={this.updateModel}
                                  variant="outline-info">
                            Update
                          </Button>
                        }
                        {this.state.isDownloadingModel &&
                          <div>
                            <Spinner animation="border" role="status" size="sm">
                              <span className="sr-only">Downloading...</span>
                            </Spinner>
                            {' '}<strong>Downloading...</strong>
                          </div>
                        }
                      </div>
                  </Alert>
                </Container>
              }
              {this.state.showModelUpdateSuccess &&
                <Container>
                  <Alert variant="success"
                         onClose={() => this.setState({ showModelUpdateSuccess: false})}
                         dismissible>
                    The <strong>{this.state.modelType}</strong> model has been updated!
                  </Alert>
                </Container>
              }
            <Tabs defaultActiveKey="camera" id="input-options" onSelect={this.handleTabSelect}
                  className="justify-content-center">
              <Tab eventKey="camera" title="Camera">
                <div id="no-webcam" ref="noWebcam">
                  <span className="camera-icon"><FaCamera /></span>
                  No camera found. <br />
                  Please use a device with a camera, or upload an image instead.
                </div>
                <div className="webcam-box-outer">
                  <div className="webcam-box-inner">
                    <video ref="webcam" autoPlay playsInline muted id="webcam"
                           width="448" height="448">
                    </video>
                  </div>
                </div>
                <div className="button-container">
                  <LoadButton
                    variant="primary"
                    size="lg"
                    onClick={this.classifyWebcamImage}
                    isLoading={this.state.isClassifying}
                    text="Begin Counting"
                    loadingText= 'Begin Exercise'
                  />
                </div>
                <div id="timer" hidden>
                </div>
                <div id="pushup_count" hidden>
                </div>
                <div id="straighten_arm"  hidden>Straighten Your Arm
                </div>
              </Tab>
            </Tabs>
            </div>
          </Collapse>
          </Fragment>
        }
      </div>
    );
  }
}
