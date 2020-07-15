# GraphFlow


Code & data accompanying the IJCAI 2020 paper ["GRAPHFLOW: Exploiting Conversation Flow with Graph Neural Networks for Conversational Machine Comprehension"](https://www.ijcai.org/Proceedings/2020/171)


## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.



### Run the model

* Download the preprocessed data from [here](https://1drv.ms/u/s!AjiSpuwVTt09gTtAGzIRsp6Py3q-?e=Yxqa7w) and put the data folder under the root directory. (Note: if you cannot access the above data, please download from [here](http://academic.hugochan.net/download/graphflow-data.zip).)

* Run the model

	```
	python main.py -config config/graphflow_dynamic_graph_coqa.yml
	```


### Prepare your own data

* Download the raw data
	
	```
	sh download.sh
	```

* Run the stanford-core-nlp script

	check out https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

* Run the preprocessing script
	
	```
	python coqa_scripts/preprocess.py -d path_to_input_data -o path_to_output_data
	```

* Annotate the data if you want to have the input passage represented as graph-structured data
	
	```
	python annotate_graphs.py -i path_to_input_data -o path_to_output_data
	```


## Reference

If you found this code useful, please consider citing the following paper:

Yu Chen, Lingfei Wu, Mohammed J. Zaki. **"Graphflow: Exploiting Conversation Flow with Graph Neural Networks for Conversational Machine Comprehension."** In *Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI 2020)*, Yokohama, Japan, Jul 11-17, 2020.


    @article{chen2019graphflow,
      title={Graphflow: Exploiting conversation flow with graph neural networks for conversational machine comprehension},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J},
      journal={arXiv preprint arXiv:1908.00059},
      year={2019}
    }
