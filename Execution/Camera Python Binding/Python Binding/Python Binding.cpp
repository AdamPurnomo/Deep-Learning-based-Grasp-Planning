// Python Binding.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "dependencies.h"

static PyObject* getpointmap(PyObject* self, PyObject* args)
{
	nxLibInitialize(true);

	NxLibItem root;

	NxLibItem camera = root[itmCameras][itmBySerialNo]["171423"];
	std::string serial = camera[itmSerialNumber].asString();

	NxLibCommand open(cmdOpen);
	open.parameters()[itmCameras] = serial;
	open.execute();

	std::ifstream file("../Temporary Data/params.json");
	if (file.is_open() && file.rdbuf()) {

		std::stringstream buffer;
		buffer << file.rdbuf();
		std::string const& fileContent = buffer.str();

		NxLibItem tmp("/tmp");
		tmp.setJson(fileContent); // Parse json file content into temporary item
		if (tmp[itmParameters].exists()) {


			camera[itmParameters].setJson(tmp[itmParameters].asJson(), true);

		}
		else {

			camera[itmParameters].setJson(tmp.asJson(), true);
		}
	}
	else {
		std::cout << "File couldn't be read!" << "\n";
		// File could not be read
	}


	NxLibCommand capture(cmdCapture);
	capture.parameters()[itmTimeout] = 10000;
	capture.execute();

	NxLibCommand cdisparity(cmdComputeDisparityMap);
	cdisparity.execute();

	NxLibCommand cpoint(cmdComputePointMap);
	cpoint.execute();

	NxLibCommand cnormals(cmdComputeNormals);
	cnormals.parameters()[itmCameras] = serial;
	cnormals.execute();

	//copy the binary data

	int width, height;
	std::vector<float> disparitymap;
	camera[itmImages][itmDisparityMap].getBinaryDataInfo(&width, &height, 0, 0, 0, 0);
	camera[itmImages][itmDisparityMap].getBinaryData(disparitymap, 0);

	std::vector<float> pointmap;
	int rwidth, rheight;
	camera[itmImages][itmPointMap].getBinaryDataInfo(&rwidth, &rheight, 0, 0, 0, 0);
	camera[itmImages][itmPointMap].getBinaryData(pointmap, 0);
	std::cout << "Width : " << rwidth << "\n";
	std::cout << "Height : " << rheight << "\n";

	NxLibCommand close(cmdClose);
	close.parameters()[itmCameras] = serial;
	close.execute();
	std::vector<float>::const_iterator it;

	PyObject* result = PyList_New(0);

	for (it = pointmap.begin(); it != pointmap.end(); it++)
	{
		PyList_Append(result, PyFloat_FromDouble(*it));
	}
	std::cout << "Point Map is successfully obtained" << "\n";
	return result;

}

static PyMethodDef CameraMethods[] = {
	{ "getpointmap", getpointmap, METH_NOARGS, "getting point map from camera" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef gpmmodule = {
	PyModuleDef_HEAD_INIT,
	"camera",
	NULL,
	-1,
	CameraMethods
};

PyMODINIT_FUNC
PyInit_getpointmap(void)
{
	return PyModule_Create(&gpmmodule);
}
