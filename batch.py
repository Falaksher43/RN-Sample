
from app import *


if __name__ == '__main__':
    # in case there were any visits stuck in processing if something crashed
    # reset them here when turning on the API

    reset_processing()

    if len(sys.argv) == 2:
        # check that parsing it goes OK
        json_params = json.loads(sys.argv[1])
        logging.info("resetting batch process with the following parameters: ")
        logging.info(str(json_params))
        # set all the parameters for all the elements
        reset_batch_processing(sys.argv[1])

    elif len(sys.argv) > 2:
        logging.error("sorry but you can only pass one parameter to app")
    else:
        # you didn't pass any params, so only set the batch begin time plz
        reset_batch_processing('{}')

    # whether it is batch or not, please run this!
    flask_config = udb.load_json("flask_config.json")
    app.run(debug=False, use_reloader=False, host=flask_config['host'], port=flask_config['port'])


