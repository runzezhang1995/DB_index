import React from 'react';
import { hot } from 'react-hot-loader';
import './DemoPage.styl';

import { Button, Row, Col, Image, Modal} from 'react-bootstrap';
import ImageGrid from './gridlayout';
import text from './text.json'
// // import  Row  from 'react-bootstrap/Row';

// // import Col from 'react-bootstrap/Col';
// // import Image from 'react-bootstrap/Image';
// import Modal from 'react-bootstrap/Modal';

const image_base_url = 'http://99.23.139.146:1551/image/';



class Recognition extends React.Component {
    constructor(props, context) {
        super(props, context);
    
        this.handleShow = this.handleShow.bind(this);
        this.handleClose = this.handleClose.bind(this);
        this.getReidResult = this.getReidResult.bind(this);

        this.state = {
            queryImageSelected: false,
            queryProcessing: false,
            resultCalculated: false,
            QImageSrc: "",
            GImageSrcs: {},
            showModal: false,
            baselineTime: 0,
            kmeansTime:0,
            baselineMem: 0,
            kmeansMem: 0
        }
    }

    onImageClicked = (e) => {
        console.log(e.target.id);
        this.setState({
            QImageSrc: e.target.id,
            queryImageSelected: true,
            GImageSrcs: {},
            resultCalculated: false
        });
        this.handleClose();

    }
    
    updateImageHolder(){
        if(this.state.queryImageSelected){
            console.log('has image');
            return(
                <Image src={image_base_url + this.state.QImageSrc} alt="image not found" srcSet="" className="query-image" fluid rounded/>
            ); 
        } else {
            console.log('has no image');
            return (
                <h6 className = "image-holder-text"> No Image Selected </h6>
            );
        }
    }


    handleClose() {
        
        this.setState({
            showModal: false,
        });
    }
    
    handleShow() {
        this.setState({
            showModal: true,
        });
    }


    getQueryImage(addr) {
        return (
        <Image src={image_base_url + addr} alt="image not found" srcSet="" className="query-image-thumbnail" id={addr} onClick={this.onImageClicked} thumbnail/>
        ); 
    }
    getReidResult(){
        console.log('clicked');
        this.setState({
            queryProcessing: true
        });

        $.post('/api/recognize/' + this.state.QImageSrc, (data, status) => {
            console.log(data)
            this.setState({
                resultCalculated: true,
                GImageSrcs: data,
                queryProcessing: false,
                baselineTime: [data.result_baseline.time_db, data.result_baseline.time_recog],
                kmeansTime:[data.result_kmeans.time_db, data.result_kmeans.time_recog],
                baselineMem: data.result_baseline.mem,
                kmeansMem: data.result_kmeans.mem
            })
        }, 'json');

    //     const image_path = this.state.QImageSrc;
    //     this.serverRequest = $.ajax({
    //         method:'POST',
    //         url: python_root_url + '/reid/recognize',
    //         dataType: 'json',
    //         data: {image_path},
    //         xhrFields: {'Access-Control-Allow-Origin': '*',
    //                     'Access-Control-Request-Method': '*',
    //                     'Access-Control-Allow-Methods':'OPTIONS, GET, POST',
    //                     'Access-Control-Allow-Headers':'Content-Type, Depth, User-Agent, X-File-Size, X-Requested-With, If-Modified-Since, X-File-Name, Cache-Control',
    //                     'withCredentials': false
    //                     }
    //     }).done((data) => {
    //         this.setState({
    //             queryProcessing: false
    //         });
    //         this.setState({
    //             resultCalculated: true,
    //             GImageSrcs: data
    //         });

    //     }).fail((jqXHR, textStatus)=>{
    //         this.setState({
    //             queryProcessing: false
    //         });
    //         console.log( "fail: " + textStatus)
    //     });
    }


    render = () => (
        <div className="main-block">
            <Row className="header-Row">
                <div className="reid-intro">
                    <h2 className="subtitle"> {text.header} </h2>
                    <p className="content"> {text.reid_intro}  </p>
                </div>
            </Row>
            <Row className="main-Row">
                <Col xs lg = "3">
                    <div className="reid-imgSlt">
                        <Button variant="outline-dark" onClick={this.handleShow} className="slt-img-btn">
                            Select Query Image
                        </Button>
                        <div className="img-holder">
                        {this.updateImageHolder()}
                        </div>
                        <h6 className="notice-text"> Your selected query is shown here </h6>
                    </div>
                </Col>
                <Col xs lg = "9">
                    <div className = "reid-result">
                        {this.state.resultCalculated? (
                            <Row>
                                    <h4 className="notice-text"> Prediction results (Blue for Correct Prediction and Red for Wrong Prediction) </h4>
                            </Row>)
                        : null}
                            
                        { this.state.queryImageSelected? 
                            ( this.state.resultCalculated? 
                                (
                                <div>    
                                    <Row>             
                                        <Col xs lg= "6">
                                            <ImageGrid imageSrcs={this.state.GImageSrcs.result_baseline.result} className="ImageGrid"/>
                                            <h4 style={{marginLeft:"60px"}}> Result From Baseline</h4> 
                                            <div className="resultInfoDiv">
                                                <p>Time For Database Query: {this.state.baselineTime[0].toFixed(2)} seconds</p>
                                                <p>Time For Recognition: {this.state.baselineTime[1].toFixed(2)} seconds</p>
                                                <p>Memory Used: {this.state.baselineMem.toFixed(2)} MB</p>
                                            </div>
                                        </Col>    
                                        <Col xs lg= "6">
                                            <ImageGrid imageSrcs={this.state.GImageSrcs.result_kmeans.result} className="ImageGrid"/>
                                            <h4 style={{marginLeft:"60px"}}> Result From Kmeans Index</h4> 
                                            <div className="resultInfoDiv">
                                                <p>Time For Database Query: {this.state.kmeansTime[0].toFixed(2)} seconds</p>
                                                <p>Time For Recognition: {this.state.kmeansTime[1].toFixed(2)} seconds</p>
                                                <p>Memory Used: {this.state.kmeansMem.toFixed(2)} MB</p>
                                            </div>
                                            

                                        </Col>   
                                    </Row>     
                                </div>
                                ) : 
                                (
                                <Button variant="outline-dark" onClick={this.getReidResult} className="slt-img-btn" disabled = {this.state.queryProcessing}>
                                    {this.state.queryProcessing? "Processing" : "Recognize"}
                                </Button>
                            )
                        ) : null}
                    </div>
                </Col>
            </Row>
            <Modal show={this.state.showModal} onHide={this.handleClose} size='lg'>
                <Modal.Header closeButton>
                    <Modal.Title>Select A Query Image</Modal.Title>
                </Modal.Header>
                    {this.generateModalBody()}
            </Modal>
        </div>
    );
    
    


    generateModalBody(){
        return(
            <Modal.Body>
                <Row>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                </Row>
                <Row>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                    <Col>
                    <div className="img-holder-thumb">
                        {this.getQueryImage(this.props.queryImageList[Math.floor(Math.random()*this.props.queryImageList.length)])}
                    </div>   
                    </Col>
                </Row>
            </Modal.Body>
        )
    }
}


export default hot(module)(Recognition);      